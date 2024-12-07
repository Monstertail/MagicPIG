# cache for grape with graph
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
class GrapeCache(Cache):
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
        self.L = L #Budget of tokens in grape
        self.recall = None
        
        self.mode = mode
        self.min_key: List[torch.Tensor] = []
        self.max_key: List[torch.Tensor] = []
        self.key_hashcode: List[torch.Tensor] = []
        self.expand_key: List[torch.Tensor] = []
        self.window = window
        self.hash_matrices: List[torch.Tensor] = []
        self.preserve_layer = 2

        #grape graphs
        self.grape_graphs: List[torch.Tensor] = []
        # selected tokens per iteration
        self.selected_tokens: List[torch.Tensor] = []

        
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

                # jinwei: build the grape graph for the layer
                self.build_grape_graph(key_states, value_states, query_states, layer_idx)
                self.init_min_max_key(layer_idx)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.prefill_tokens += key_states.shape[2]

                # jinwei: build the grape graph for the layer
                self.build_grape_graph(key_states, value_states, query_states, layer_idx)
                self.init_min_max_key(layer_idx)

                return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            # jinwei: update min max key 
            self.update_min_max_key(key_states, layer_idx)
            # jinwei: sparse attention based on grape graph
            if layer_idx > 2:
                # get union indices of all heads in the previous selected tokens
                selected_tokens = self.selected_tokens[layer_idx-1] # dim: [batch, num_heads, 1, num_activate_tokens]     
                # assert the batch size is 1
                assert selected_tokens.shape[0] == 1, "batch size should be 1 for current grape implementation!"
                # for each query in the batch, we get the union of selected tokens
                # Extract dimensions
                batch_size, num_heads, num_queries, num_activate_tokens = selected_tokens.shape
                # Reshape to flatten heads and tokens
                # Remove the batch size dimension since it is 1
                flattened_indices = selected_tokens.squeeze(0).view(-1)  # Shape: [num_heads * num_activate_tokens]
                # Perform unique operation to compute the union of indices across heads
                # Shape: [num_unique_tokens]
                union_indices = flattened_indices.unique()
                # union_indices now contains the unified token indices for all heads
                # grape selection of tokens in range [:-window size] for each head
                selected_indices= self.grape_selection(union_indices_before=union_indices, layer_idx=layer_idx, middle_budget=self.L, query_states=query_states) # dim: [batch, num_heads, 1, num_activate_tokens]
                assert selected_indices.shape[-1] == self.L, "selected tokens should be equal to the budget!"
                self.selected_tokens.append(selected_indices)

                # get the window size of the latest tokens. 
                # make sure the selected tokens from grape are between [:-window_size]
                window_size = self.window
                sequence_length = self.get_seq_length(layer_idx)
                # Get the scalar threshold
                threshold = sequence_length - window_size
                # Compare with broadcasting
                assert (selected_indices.max(dim=-1).values < threshold).all(), \
                    "Middle selected tokens should be before the local window!"
                
                # get the selected key and value states
                return_key = repeat_kv(self.key_cache[layer_idx], self.num_qh // self.num_kh)
                return_value = repeat_kv(self.value_cache[layer_idx], self.num_qh // self.num_kh)
                # get the key and value states from the local window
                # Get the local window indices

                local_window_start = max(sequence_length - window_size, 0)  # Ensure the start is non-negative
                local_window_indices = torch.arange(local_window_start, sequence_length, device=selected_indices.device)
                # Expand local_window_indices to match selected_indices dimensions: [batch, num_heads, 1, local_window_size]
                local_window_indices = local_window_indices.view(1, 1, 1, -1).expand(
                    selected_indices.shape[0], selected_indices.shape[1], 1, -1
                )

                # Merge selected_indices and local_window_indices
                merged_indices = torch.cat([selected_indices, local_window_indices], dim=-1)

                # Gather keys and values using the merged indices
                merged_key = return_key.gather(dim=-2, index=merged_indices)
                merged_value = return_value.gather(dim=-2, index=merged_indices)

                # Compute attention with the merged keys and values
                attn_weights = torch.matmul(query_states, merged_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, merged_value)

                return attn_output
                
            
            
            else: # full attention for layer 0-2
                return_key = repeat_kv(self.key_cache[layer_idx], self.num_qh // self.num_kh)
                return_value = repeat_kv(self.value_cache[layer_idx], self.num_qh // self.num_kh)
                        
                attn_weights = torch.matmul(query_states, return_key.transpose(2, 3)) / math.sqrt(self.head_dim)
                #jinwei: select topk tokens for layer 2
                if layer_idx == 2:
                    num_activate_tokens = self.L
                    topk_indices = attn_weights.topk(k=num_activate_tokens, dim=-1).indices # dim: [batch, num_heads, 1, num_activate_tokens]
                    self.selected_tokens.append(topk_indices)

                elif layer_idx == 0:
                    # clear the selected tokens
                    self.selected_tokens = []
                    self.selected_tokens.append(None)
                else: # layer_idx == 1
                    self.selected_tokens.append(None)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32) 
                attn_weights = attn_weights.to(query_states.dtype)
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
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "GrapeCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def build_page(self, layer_idx: int):
        pass
    
    
    # jinwei: build the grape graph for the layer
    def build_grape_graph(self,  key_states: torch.Tensor, value_states: torch.Tensor,query_states: torch.Tensor,layer_idx: int):
        # for layer_idx<2, we do not build the graph and leave it empty
        if layer_idx < 2:
            if len(self.grape_graphs) <= layer_idx:
                self.grape_graphs.append(None)
            
        # for layer_idx>=2, we build the graph
        else:
            qk = torch.matmul(query_states, key_states.transpose(2,3)) #dim: [batch, num_heads, num_q, num_k]
            # get the max indice for each key
            q_indices = qk.argmax(dim=-2)
            # get the topk indices for the query-key similarity matrix
            # number of neighbors kn is set to 2.
            # todo(jinwei): set kn to self.K in the future
            kn = 2
            topkn_indices = qk.topk(k=kn, dim=-1).indices
            grape_neighbors = topkn_indices[q_indices] # dim: [batch, num_heads, num_k, kn]
            # build the graph
            if len(self.grape_graphs) <= layer_idx:
                self.grape_graphs.append(grape_neighbors)
            else:
                self.grape_graphs[layer_idx] = grape_neighbors
        
    def update_grape_graph(self, key_states: torch.Tensor, value_states: torch.Tensor, query_states: torch.Tensor, layer_idx: int):
        pass

    def init_min_max_key(self, layer_idx: int):
        # key dim: [batch, num_heads, num_keys, head_dim]
        k_cache = self.key_cache[layer_idx]
        self.min_key.append(k_cache.min(dim=-1).values)
        self.max_key.append(k_cache.max(dim=-1).values)

    def  update_min_max_key(self, key_states: torch.Tensor, layer_idx: int):
        # min_key dim: [batch, num_heads, head_dim]
        new_min = key_states.min(dim=-1).values  # [batch, num_heads, num_keys]
        new_max = key_states.max(dim=-1).values  # [batch, num_heads, num_keys]
        self.min_key[layer_idx] = torch.cat([self.min_key[layer_idx], new_min], dim=-1)  
        self.max_key[layer_idx] = torch.cat([self.max_key[layer_idx], new_max], dim=-1)  

    def grape_selection(
        self, union_indices_before: torch.Tensor, layer_idx: int, middle_budget: int, query_states: torch.Tensor
    ) -> torch.Tensor:  # Output the expanded tokens with dim: [batch=1, num_heads, 1, middle_budget]
        # union_indices_before: dim: [num_unique_tokens]
        grape_graph = self.grape_graphs[layer_idx]  # dim: [batch, num_heads, num_k, kn]
        key_len_grape = grape_graph.shape[-2]

        # Initialize the selected tokens to -1
        selected_indices = torch.zeros(
            1, grape_graph.shape[1], 1, middle_budget, device=grape_graph.device, dtype=query_states.dtype  
        )

        # For each head, expand the key candidates based on the graph
        for head in range(grape_graph.shape[1]):
            # Get the key neighbor graph for this head
            key_neighbor_graph = grape_graph[0, head]  # dim: [num_k, kn]

            # Identify neighbors of union_indices_before in the key_neighbor_graph
            expanded_indices = []
            for idx in union_indices_before:
                expanded_indices.extend(key_neighbor_graph[idx].tolist())
            expanded_indices = torch.tensor(expanded_indices, device=grape_graph.device, dtype=query_states.dtype).unique()

            # Filter the expanded indices to ensure they are before the local window
            window_size = self.window
            sequence_length = self.get_seq_length(layer_idx)
            threshold = sequence_length - window_size
            filtered_indices = expanded_indices[expanded_indices < threshold]

            if len(filtered_indices) >= middle_budget:
                # Fine-grained selection of tokens by quantized dot product
                selected_indices[0, head, 0, :] = self.quantized_dot_product(
                    query_states=query_states[0, head],
                    key_states=self.key_cache[layer_idx][0, head],
                    indices_before_selection=filtered_indices,
                    budget=middle_budget,
                    layer_idx=layer_idx
                )
            else:
                # Expand the indices from the end of tokens in [:-window_size]
                # Ensure the number of selected tokens equals `middle_budget`
                additional_indices = torch.arange(
                    max(0, threshold - (middle_budget - len(filtered_indices))),
                    threshold,
                    device=grape_graph.device,
                    dtype=torch.long
                )
                selected_indices[0, head, 0, :len(filtered_indices)] = filtered_indices
                selected_indices[0, head, 0, len(filtered_indices):] = additional_indices[:middle_budget - len(filtered_indices)]

        return selected_indices
    # todo(jinwei): implement the quantized dot product for fine-grained selection of tokens
    def quantized_dot_product(
        self, query_states: torch.Tensor, key_states: torch.Tensor, indices_before_selection: torch.Tensor, budget: int, layer_idx: int
    ) -> torch.Tensor:
        # Get the key states for the selected indices
        key_states_before_selection = key_states[:, indices_before_selection, :]  # Shape: [1, len(indices_before_selection), head_dim]
        
        # Compute the dot product between query and selected keys
        qk = torch.matmul(query_states.unsqueeze(1), key_states_before_selection.transpose(1, 2))  # dim: [1, 1, len(indices_before_selection)]
        
        # Top-k selection based on dot product
        topk_indices = qk.squeeze(1).topk(k=budget, dim=-1).indices  # Shape: [budget]
        
        # Map the top-k indices back to the original indices
        return indices_before_selection[topk_indices]
        