# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.mapping import Mapping
from .config import Exaone4Config


def load_exaone4_weights_from_hf_model(
    hf_model_dir: Union[str, Path],
    config: Exaone4Config,
    model
) -> Dict[str, torch.Tensor]:
    # Load HuggingFace model
    print(f"Loading HuggingFace Exaone 4.0 model from {hf_model_dir}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="cpu"
    )
    
    # Get state dict
    hf_state_dict = hf_model.state_dict()
    
    # Initialize weights dictionary
    weights = {}
    
    # Convert embedding weights
    print("Converting embedding weights...")
    weights["transformer.vocab_embedding.weight"] = hf_state_dict["model.embed_tokens.weight"]
    
    # Convert decoder layer weights
    print("Converting decoder layer weights...")
    for layer_idx in range(config.num_hidden_layers):
        convert_decoder_layer_weights(
            hf_state_dict=hf_state_dict,
            weights=weights,
            layer_idx=layer_idx,
            config=config
        )
    
    # Convert final layer norm
    print("Converting final layer norm...")
    weights["transformer.ln_f.weight"] = hf_state_dict["model.norm.weight"]
    
    print("Converting language model head...")
    if "lm_head.weight" in hf_state_dict:
        weights["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    else:
        weights["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]
        print("Using tied embeddings for lm_head.weight")
    
    print(f"Successfully converted {len(weights)} weight tensors")
    return weights


def convert_decoder_layer_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    layer_idx: int,
    config: Exaone4Config
):
    """
    Convert single decoder layer weights from HF to TensorRT-LLM format.
    
    Handles Exaone 4.0 specific components:
    - QKV projections with different head counts
    - QK LayerNorm weights
    - Post-norm architecture weights
    - Gated MLP weights
    """
    hf_prefix = f"model.layers.{layer_idx}"
    trt_prefix = f"transformer.layers.{layer_idx}"
    
    # === Attention Weights ===
    convert_attention_weights(hf_state_dict, weights, hf_prefix, trt_prefix, config, layer_idx)
    
    # === MLP Weights ===
    convert_mlp_weights(hf_state_dict, weights, hf_prefix, trt_prefix, config)
    
    # === Normalization Weights (Post-norm specific) ===
    convert_normalization_weights(hf_state_dict, weights, hf_prefix, trt_prefix, config)


def convert_attention_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    hf_prefix: str,
    trt_prefix: str,
    config: Exaone4Config,
    layer_idx: int
):
    
    # Get original weight tensors
    q_weight = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
    k_weight = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"] 
    v_weight = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]
    
    # Validate dimensions
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_size = hidden_size // num_heads
    
    assert q_weight.shape == (hidden_size, hidden_size), f"Unexpected q_proj shape: {q_weight.shape}"
    assert k_weight.shape == (num_kv_heads * head_size, hidden_size), f"Unexpected k_proj shape: {k_weight.shape}"
    assert v_weight.shape == (num_kv_heads * head_size, hidden_size), f"Unexpected v_proj shape: {v_weight.shape}"
    
    # Handle tensor parallelism
    tp_size = config.mapping.tp_size
    tp_rank = config.mapping.tp_rank
    
    if tp_size > 1:
        # Split heads across tensor parallel ranks
        heads_per_rank = num_heads // tp_size
        kv_heads_per_rank = num_kv_heads // tp_size
        
        q_start = tp_rank * heads_per_rank * head_size
        q_end = (tp_rank + 1) * heads_per_rank * head_size
        
        kv_start = tp_rank * kv_heads_per_rank * head_size
        kv_end = (tp_rank + 1) * kv_heads_per_rank * head_size
        
        q_weight = q_weight[q_start:q_end, :]
        k_weight = k_weight[kv_start:kv_end, :]
        v_weight = v_weight[kv_start:kv_end, :]
    
    # Store QKV weights
    weights[f"{trt_prefix}.attention.qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)
    
    # Output projection
    o_weight = hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"]
    if tp_size > 1:
        o_start = tp_rank * (hidden_size // tp_size)
        o_end = (tp_rank + 1) * (hidden_size // tp_size)
        o_weight = o_weight[:, o_start:o_end]
    weights[f"{trt_prefix}.attention.dense.weight"] = o_weight
    
    # QK LayerNorm weights (Exaone 4.0 specific) - Safe handling
    q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
    k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
    
    if q_norm_key in hf_state_dict and k_norm_key in hf_state_dict:
        q_norm_weight = hf_state_dict[q_norm_key]
        k_norm_weight = hf_state_dict[k_norm_key]
        
        if tp_size > 1:
            q_norm_weight = q_norm_weight[q_start:q_end]
            k_norm_weight = k_norm_weight[kv_start:kv_end]
        
        weights[f"{trt_prefix}.attention.q_layernorm.weight"] = q_norm_weight
        weights[f"{trt_prefix}.attention.k_layernorm.weight"] = k_norm_weight
        pass
    else:
        pass


def convert_mlp_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    hf_prefix: str, 
    trt_prefix: str,
    config: Exaone4Config
):
    """Convert MLP weights for gated architecture."""
    
    # Gate and up projections
    gate_weight = hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"]
    up_weight = hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"]
    
    # Handle tensor parallelism
    tp_size = config.mapping.tp_size
    tp_rank = config.mapping.tp_rank
    
    if tp_size > 1:
        intermediate_size = config.intermediate_size
        split_size = intermediate_size // tp_size
        start_idx = tp_rank * split_size
        end_idx = (tp_rank + 1) * split_size
        
        gate_weight = gate_weight[start_idx:end_idx, :]
        up_weight = up_weight[start_idx:end_idx, :]
    
    # GatedMLP: fc contains up_weight only, gate is separate
    weights[f"{trt_prefix}.mlp.gate.weight"] = gate_weight  
    weights[f"{trt_prefix}.mlp.fc.weight"] = up_weight
    
    # Down projection
    down_weight = hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"]
    if tp_size > 1:
        down_weight = down_weight[:, start_idx:end_idx]
    weights[f"{trt_prefix}.mlp.proj.weight"] = down_weight


def convert_normalization_weights(
    hf_state_dict: Dict[str, torch.Tensor],
    weights: Dict[str, torch.Tensor],
    hf_prefix: str,
    trt_prefix: str, 
    config: Exaone4Config
):
    
    if config.use_post_norm:
        # Post-attention layer norm (Exaone 4.0 specific)
        post_attn_weight = hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]
        weights[f"{trt_prefix}.post_attention_layernorm.weight"] = post_attn_weight
        
        # Post-feedforward layer norm (Exaone 4.0 specific)
        post_ff_weight = hf_state_dict[f"{hf_prefix}.post_feedforward_layernorm.weight"]
        weights[f"{trt_prefix}.post_feedforward_layernorm.weight"] = post_ff_weight
    else:
        # Standard pre-normalization (fallback)
        input_norm_weight = hf_state_dict[f"{hf_prefix}.input_layernorm.weight"]
        post_attn_weight = hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]
        
        weights[f"{trt_prefix}.input_layernorm.weight"] = input_norm_weight
        weights[f"{trt_prefix}.post_attention_layernorm.weight"] = post_attn_weight


# Quantization support (similar to Gemma)
class QuantizeModifiers:
    """Quantization modifiers for Exaone 4.0"""
    pass


class Weights:
    """Weight container for Exaone 4.0"""
    pass


def non_modelopt_quantize_if_needed(config, weights):
    """Apply non-ModelOpt quantization if needed"""
    # Implementation would go here for non-ModelOpt quantization
    # For now, return weights as-is
    return weights
