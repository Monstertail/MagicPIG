from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple
import torch
import math
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn.functional as F

def repeat_kv2(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, chunk_size, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :, :].expand(batch, num_key_value_heads, n_rep, slen, chunk_size, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, chunk_size, head_dim)
class QuestCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, K, L, mode = "anns", window = 64) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.selected_key_cache: List[torch.Tensor] = []
        self.selected_value_cache: List[torch.Tensor] = []
        self.unselected_key_cache: List[torch.Tensor] = []
        self.unselected_value_cache: List[torch.Tensor] = []
        self.prefill_tokens = 0
        self.sampling_prob: torch.Tensor = None
        self.kernel_size = 5
        self.interleave = 8
        self.hash_matrix = None
        self.num_qh = None
        self.num_kh = None
        self.head_dim = None
        self.K = K #Page Size
        self.L = L #Budget 
        self.recall = None
        self.mode = mode
        self.min_key: List[torch.Tensor] = []
        self.max_key: List[torch.Tensor] = []
        self.key_hashcode: List[torch.Tensor] = []
        self.expand_key: List[torch.Tensor] = []
        self.window = window
        self.hash_matrices: List[torch.Tensor] = []
        self.preserve_layer = 2
        # jinwei: I implement a function to update page to keep the window size nearest tokens rather than the last 'window size' tokens in prompt.
        self.incomplete_key_buffer: List[torch.Tensor] = []
        self.incomplete_value_buffer: List[torch.Tensor] = []
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        random_sparse: float = 1.0,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        
        

        # Update the cache
        if key_states.shape[2] > 1:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.prefill_tokens += key_states.shape[2]
                self.num_qh = query_states.shape[1]
                self.num_kh = key_states.shape[1]
                self.head_dim = key_states.shape[-1]  
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.prefill_tokens += key_states.shape[2]
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.selected_key_cache[layer_idx] = torch.cat([self.selected_key_cache[layer_idx], key_states], dim=-2)
            self.selected_value_cache[layer_idx] = torch.cat([self.selected_value_cache[layer_idx], value_states], dim=-2)
            #jinwei: update page
            self.update_page(layer_idx)
            num_activate_chunks = int(self.L)
            if layer_idx >= 2:
                if num_activate_chunks > 0:
                    min_cache = repeat_kv(self.min_key[layer_idx], self.num_qh // self.num_kh)
                    max_cache = repeat_kv(self.max_key[layer_idx], self.num_qh // self.num_kh)

                    #jinwei(todo):add min and max of key buffer to min and max cache for selection. now only use the min and max of selected key cache without buffer.
                    

                    min_value = min_cache * query_states
                    max_value = max_cache * query_states

                    heuristic = torch.max(min_value, max_value)
                    heuristic = heuristic.sum(dim=-1)
                    
                    topk_chunk = heuristic.topk(k=num_activate_chunks, dim=-1).indices
                    topk_chunk = topk_chunk[:,:,:,None,None].expand(-1,-1,-1,self.K, self.head_dim)
                    
                    unselected_key_cache = repeat_kv2(self.unselected_key_cache[layer_idx], self.num_qh // self.num_kh)
                    unselected_value_cache = repeat_kv2(self.unselected_value_cache[layer_idx], self.num_qh // self.num_kh)
                    
                    k = unselected_key_cache.gather(dim=2, index=topk_chunk)
                    v = unselected_value_cache.gather(dim=2, index=topk_chunk)
                    
                    k = k.reshape(1, self.num_qh, num_activate_chunks * self.K, self.head_dim)
                    v = v.reshape(1, self.num_qh, num_activate_chunks * self.K, self.head_dim)
                    
                    select_k = repeat_kv(self.selected_key_cache[layer_idx], self.num_qh // self.num_kh)
                    select_v = repeat_kv(self.selected_value_cache[layer_idx], self.num_qh // self.num_kh)
                    k = torch.cat([k, select_k], dim=-2)
                    v = torch.cat([v, select_v], dim=-2)

                    
                    attn_weights = torch.matmul(query_states, k.transpose(2, 3)) / math.sqrt(self.head_dim)
                    
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
                    attn_output = torch.matmul(attn_weights, v)
                    
                    return attn_output
                
                else:
                    return_key = self.selected_key_cache[layer_idx]
                    return_value = self.selected_value_cache[layer_idx]
                    
                    return_key = repeat_kv(return_key, self.num_qh // self.num_kh)
                    return_value = repeat_kv(return_value, self.num_qh // self.num_kh)
                        
                    attn_weights = torch.matmul(query_states, return_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                        
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
                    attn_output = torch.matmul(attn_weights, return_value)
                    return attn_output
            else:
                return_key = repeat_kv(self.key_cache[layer_idx], self.num_qh // self.num_kh)
                return_value = repeat_kv(self.value_cache[layer_idx], self.num_qh // self.num_kh)
                        
                attn_weights = torch.matmul(query_states, return_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                        
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
                attn_output = torch.matmul(attn_weights, return_value)
                return attn_output

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "QuestCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def build_page(self, layer_idx: int):
        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]

        seq_len = k_cache.shape[2]
        self.num_chunks = (seq_len-64) // self.K
        
        paged_length = self.num_chunks * self.K
        self.selected_key_cache.append(k_cache[...,paged_length:,:])
        self.selected_value_cache.append(v_cache[...,paged_length:,:])


        self.unselected_key_cache.append(k_cache[...,:paged_length,:].reshape(1, self.num_kh, self.num_chunks, self.K, self.head_dim))
        self.unselected_value_cache.append(v_cache[...,:paged_length,:].reshape(1, self.num_kh, self.num_chunks, self.K, self.head_dim))

        unselected_key_cache = k_cache[...,:paged_length,:]
        
        unselected_key_cache = k_cache[...,:paged_length,:].reshape(1, self.num_kh, self.num_chunks, self.K, self.head_dim)

        self.min_key.append(unselected_key_cache.min(dim=-2).values)
        self.max_key.append(unselected_key_cache.max(dim=-2).values)
    
    # jinwei: I implement a function to update page to keep the window size nearest tokens rather than the last 'window size' tokens in prompt.
    def update_page(self, layer_idx: int):
        """
        Updates the selected_key_cache for a specific layer, ensuring it retains exactly `window_size` tokens.
        Evicts excess tokens into a buffer. When the buffer accumulates enough tokens for a full page,
        they are added to unselected_key_cache. Incomplete pages are NOT included in min_key and max_key calculations.
        """
        self.ensure_buffer_initialized(layer_idx)
        # Access the current selected cache
        k_cache = self.selected_key_cache[layer_idx]
        v_cache = self.selected_value_cache[layer_idx]

        seq_len = k_cache.shape[2]  # Current sequence length

        # Ensure window_size >= page_size
        if self.window < self.K:
            raise ValueError("window_size must be greater than or equal to page_size (K).")

        # Check if the selected cache exceeds the window size
        if seq_len > self.window:
            # Split into valid window and excess tokens
            excess_tokens = seq_len - self.window
            excess_key_cache = k_cache[..., :excess_tokens, :]
            excess_value_cache = v_cache[..., :excess_tokens, :]
            remaining_key_cache = k_cache[..., excess_tokens:, :]
            remaining_value_cache = v_cache[..., excess_tokens:, :]

            # Update selected cache to retain only the valid window
            self.selected_key_cache[layer_idx] = remaining_key_cache
            self.selected_value_cache[layer_idx] = remaining_value_cache

            # Add evicted tokens to buffer
            if self.incomplete_key_buffer[layer_idx] is not None:
                excess_key_cache = torch.cat([self.incomplete_key_buffer[layer_idx], excess_key_cache], dim=-2)
                excess_value_cache = torch.cat([self.incomplete_value_buffer[layer_idx], excess_value_cache], dim=-2)
            else:
                # Initialize the buffer with evicted tokens
                self.incomplete_key_buffer[layer_idx] = excess_key_cache
                self.incomplete_value_buffer[layer_idx] = excess_value_cache
            # Split buffer into full pages and remaining tokens
            num_chunks = excess_key_cache.shape[-2] // self.K
            remaining_tokens = excess_key_cache.shape[-2] % self.K

            if num_chunks > 0:
                full_key_pages = excess_key_cache[..., :num_chunks * self.K, :].reshape(
                    1, self.num_kh, num_chunks, self.K, self.head_dim
                )
                full_value_pages = excess_value_cache[..., :num_chunks * self.K, :].reshape(
                    1, self.num_kh, num_chunks, self.K, self.head_dim
                )

                # Append full pages to unselected cache
                if layer_idx < len(self.unselected_key_cache):
                    self.unselected_key_cache[layer_idx] = torch.cat(
                        [self.unselected_key_cache[layer_idx], full_key_pages], dim=2
                    )
                    self.unselected_value_cache[layer_idx] = torch.cat(
                        [self.unselected_value_cache[layer_idx], full_value_pages], dim=2
                    )
                else:
                    raise ValueError("unselected_key_cache and unselected_value_cache must be initialized before updating the cache.")

                # Update min_key and max_key for the full pages
                buffer_min = full_key_pages.min(dim=-2).values
                buffer_max = full_key_pages.max(dim=-2).values
                if layer_idx < len(self.min_key):
                    # Dynamically determine the dimensions of self.min_key[layer_idx]
                    _, num_kh, num_existing_chunks, head_dim = self.min_key[layer_idx].shape
                    # Reshape buffer_min and buffer_max to align with self.min_key[layer_idx]
                    buffer_min = buffer_min.reshape(1, num_kh, 1, head_dim)  # Add the chunk dimension
                    buffer_max = buffer_max.reshape(1, num_kh, 1, head_dim)
                   # Concatenate along the chunk dimension (dim=2)
                    self.min_key[layer_idx] = torch.cat([self.min_key[layer_idx], buffer_min], dim=2)
                    self.max_key[layer_idx] = torch.cat([self.max_key[layer_idx], buffer_max], dim=2)
                else:
                    raise ValueError("min_key and max_key must be initialized before updating the cache.")

            # Store remaining tokens in the buffer
            self.incomplete_key_buffer[layer_idx] = (
                excess_key_cache[..., num_chunks * self.K:, :] if remaining_tokens > 0 else None
            )
            self.incomplete_value_buffer[layer_idx] = (
                excess_value_cache[..., num_chunks * self.K:, :] if remaining_tokens > 0 else None
            )
    
    def ensure_buffer_initialized(self, layer_idx: int):
        """
        Ensures that the incomplete_key_buffer and incomplete_value_buffer have been initialized
        for the given layer_idx.
        """
        while len(self.incomplete_key_buffer) <= layer_idx:
            self.incomplete_key_buffer.append(None)

        while len(self.incomplete_value_buffer) <= layer_idx:
            self.incomplete_value_buffer.append(None)