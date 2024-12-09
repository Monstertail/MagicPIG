import math
import os
from typing import List, Optional, Tuple, Union
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
LlamaAttention, LlamaRMSNorm, 
LlamaRotaryEmbedding, 
LlamaDynamicNTKScalingRotaryEmbedding, 
LlamaLinearScalingRotaryEmbedding,
apply_rotary_pos_emb,
rotate_half,
repeat_kv,
LlamaMLP)
from pos_cache import PosCache
from oraclesampling_cache import OSCache
import json
import logging
import requests
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_pos import LlamaForCausalLM
model_kwargs = {"attn_implementation": "eager"}
name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
model :LlamaForCausalLM = LlamaForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, **model_kwargs)
model.config.K = 64
model.config.L = 4
model.config.window = 2
model.config.resample = False
model.config.sample_layer = 1
model.config.resample_layer = 12
model.config.cache_mode = "pos"
model.eval()
model.select_kv(False)
prompt = "Cat is black. Dog is red. Bird is yellow. Sun is blue. Water is grey. Flowers are green. What is the color of the sun?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
seq_len = inputs["input_ids"].shape[1]
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
past_key_values = None
with torch.inference_mode():
    output = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
    print(output.past_key_values.to_legacy_cache()[0][0].shape)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, max_length=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
