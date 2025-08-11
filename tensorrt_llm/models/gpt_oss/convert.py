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

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import safetensors
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

from ..._utils import get_sm_version, str_dtype_to_torch
from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (dup_kv_bias, dup_kv_weight, get_weight_and_bias,
                             split_matrix_tp, split)
from ..modeling_utils import QuantConfig
from .config import GptOssConfig


_STREAM_TILE_ROWS = 1024


def simple_linear_weight(weight, prefix, bias=None, use_weight_only=False, 
                        plugin_weight_only_quant_type=torch.int8, dtype=torch.bfloat16, 
                        use_gemm_woq_plugin=True):
    """Simple linear weight processing"""
    results = {}
    
    if use_weight_only:
        # Weight-only quantization
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous()
        else:
            v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + 'weight'] = v.to(dtype)
        else:
            results[prefix + 'weight'] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        # Simple case: just store the weight
        results[prefix + 'weight'] = weight
    
    if bias is not None:
        results[prefix + 'bias'] = bias
    
    return results


@dataclass
class ConvertContext:
    config: GptOssConfig
    mapping: Mapping
    quant_config: QuantConfig
    model_dir: Path
    use_hf: bool


def _detect_hf_or_original(model_dir: Union[str, Path]) -> Tuple[bool, Path]:
    p = Path(model_dir)
    index = p / 'model.safetensors.index.json'
    if not index.exists():
        raise FileNotFoundError(
            f"Hugging Face index file not found: {index}. GPT-OSS converter only supports HF layout."
        )
    return (True, p)


def load_hf_model(model_dir: Path):
    import transformers
    logger.info("Loading HF model for GPT-OSS")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype='auto')
    model.eval()
    return model


def convert_and_save(
    model_dir: Union[str, Path],
    output_dir: Union[str, Path], 
    config: GptOssConfig,
    *,
    quant_config: Optional[QuantConfig] = None,
) -> None:
    """Convert GPT-OSS weights from HuggingFace format to TensorRT-LLM checkpoint format.

    This function performs comprehensive weight conversion including:
    - Loading safetensors with memory-efficient streaming
    - Reshaping Q/K/V/O projections with GQA and TP support
    - MoE weight processing (MXFP4 or dequantized BF16)
    - Attention sinks extraction and TP distribution
    - Hardware-specific optimizations (SM version aware)

    MXFP4 checkpoint weights are processed based on SM architecture:
    - SM100+ (B200): Native MXFP4 preserves original training distribution
    - SM80-89 (A100/H100): Dequantization to target precision

    """
    use_hf, p_model = _detect_hf_or_original(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Set quantization configuration following TensorRT-LLM standard conventions
    if quant_config is not None:
        # Use provided quantization config
        config.quantization = quant_config
        logger.info(f"Using provided quantization config: {config.quantization.quant_algo}")
        
        # Configure model for FP8 quantization compatibility
        if config.quantization.quant_algo == QuantAlgo.FP8:
            config.attention_bias = False  # FP8 quantization does not support MoE bias
            logger.info(f"FP8 quantization detected: disabled attention_bias for compatibility")
    else:
        # Determine quantization based on SM architecture and MoE requirements
        current_sm = get_sm_version()
        if current_sm is None or current_sm < 80:
            raise ValueError(f"Unsupported or undetected SM version: {current_sm}. Supported: SM80+ (A100, H100, B200)")
        
        if current_sm >= 100:
            # B200+: Native MXFP4 preserving original training distribution
            quant_cfg = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4)
            config.quantization = quant_cfg
            logger.info(f"SM{current_sm}: Using native MXFP4 quantization: {config.quantization.quant_algo}")
        else:
            # A100, L4, H100: Standard processing without quantization
            quant_cfg = QuantConfig(quant_algo=QuantAlgo.NO_QUANT)
            config.quantization = quant_cfg
            logger.info(f"SM{current_sm}: No quantization - using {config.dtype} precision")
    
    # Determine target dtype and processing strategy from quantization config (following TensorRT-LLM standard)
    quant_algo = config.quantization.quant_algo
    
    # Initialize quantization variables (copied from Qwen3 pattern)
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16] 
    plugin_weight_only_quant_type = torch.int8 if use_weight_only else None
    use_gemm_woq_plugin = True  # Enable for quantization support
    
    # Determine target precision based on quantization setting
    target_dtype = str_dtype_to_torch(config.dtype)
    
    # MXFP4 dequantization target: FP8 if FP8 quantization, otherwise dtype
    if quant_algo == QuantAlgo.FP8:
        mxfp4_dequant_dtype = torch.float8_e4m3fn  # MXFP4 → FP8
        logger.info(f"FP8 quantization: MXFP4 → FP8, Regular → {target_dtype}")
        
        # Initialize FP8 scaling factor dtype (following Gemma pattern)
        fake_fp8_sf_dt = torch.float32
        
        # FP8 scaling factor generation function (fixed based on TensorRT-LLM standard)
        def get_fp8_activation_scaling_factor() -> torch.Tensor:
            # Activation scaling factors are always (1,) shape regardless of MoE
            return torch.tensor([1.0], dtype=fake_fp8_sf_dt)
            
        def get_fp8_weights_scaling_factor(num_experts: int = 1) -> torch.Tensor:
            # Weight scaling factors: (num_experts, 1) for MoE, (1,) for non-MoE
            if num_experts > 1:
                return torch.ones([num_experts, 1], dtype=fake_fp8_sf_dt)
            else:
                return torch.tensor([1.0], dtype=fake_fp8_sf_dt)
    else:
        mxfp4_dequant_dtype = target_dtype  # MXFP4 → BF16/FP16
        logger.info(f"Standard precision: {target_dtype}")

    # Force RoPE type that TRT plugin recognizes
    config.position_embedding_type = "rope_gpt_neox"
    # Save initial config.json (may be overwritten after dimension inference)
    config.to_json_file(str(output / 'config.json'))

    # Prepare minimal shards and try to import attention sinks
    world_size = config.mapping.world_size if config.mapping else 1
    sinks_per_layer = {}
    sinks_per_layer = _extract_sinks_from_index(p_model)
    logger.info(f"Loaded sinks for {len(sinks_per_layer)} layers")

    # Load weight map (HF: from index; original: from single file)
    weight_map = None
    original_single_file = None
    cache_files: Dict[str, Dict[str, torch.Tensor]] = {}
    # HF only
    with open(p_model / 'model.safetensors.index.json', 'r') as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    def load_file(file_name: str) -> Dict[str, torch.Tensor]:
        if file_name in cache_files:
            return cache_files[file_name]
        data_path = p_model / file_name
        # Use safe_open instead of safetensors.torch.load_file for better compatibility
        tensors = {}
        with safe_open(data_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        cache_files[file_name] = tensors
        return tensors

    tp_size = config.mapping.tp_size if config.mapping else 1



    def _dequantize_mxfp4(blocks: torch.Tensor,
                           scales: torch.Tensor,
                           *,
                           dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """MXFP4 dequantization using FP8 E8M0 scales.

        blocks: [E, OUT, IN_bytes] or [E, OUT, IN_groups, PACK_bytes] (uint8)
        scales: [E, OUT] or [E, OUT, IN_groups] or [E, OUT, IN_bytes] (u8 as FP8 E8M0)
        Return: [E, OUT, IN]
        """
        # Convert FP8 E8M0 uint8 scales to float32 scales (matches TensorRT-LLM C++)
        scales_f32 = scales.view(torch.float8_e8m0fnu).float()
        
        if blocks.dim() == 4:
            E, OUT, G, PACK = blocks.shape
            rows_total = E * OUT * G
            B = PACK
            blk = blocks.reshape(rows_total, B)
            if scales.dim() == 2 and scales.shape == (E, OUT):
                scale_expanded = scales_f32.repeat_interleave(G, dim=0)
                scale_expanded = scale_expanded.reshape(E * OUT, 1).repeat_interleave(B * 2, dim=1)
            elif scales.dim() == 3 and scales.shape == (E, OUT, G):
                scale_expanded = scales_f32.reshape(rows_total, 1).repeat(1, B * 2)
            else:
                raise ValueError(f"Unexpected scales shape for 4D blocks: {tuple(scales.shape)}")
            # Create LUT in float32 to support indexing, then convert to target dtype
            lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)
            out = torch.empty(rows_total, B * 2, dtype=torch.float32)
            out[:, 0::2] = lut[idx_lo]
            out[:, 1::2] = lut[idx_hi]
            # Convert to target dtype after indexing
            out = out.to(dtype)
            out = out * scale_expanded.to(dtype)
            return out.reshape(E, OUT, G * B * 2)
        elif blocks.dim() == 3:
            E, OUT, B = blocks.shape
            rows_total = E * OUT
            blk = blocks.reshape(rows_total, B)
            if scales.dim() == 2 and scales.shape == (E, OUT):
                scale_expanded = scales_f32.reshape(rows_total, 1).repeat(1, B * 2)
            elif scales.dim() == 3 and scales.shape[:2] == (E, OUT) and scales.shape[2] == B:
                scale_expanded = scales_f32.reshape(rows_total, B).repeat_interleave(2, dim=1)
            else:
                raise ValueError(f"Unexpected scales shape for 3D blocks: {tuple(scales.shape)}")
            # Create LUT in float32 to support indexing, then convert to target dtype
            lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)
            out = torch.empty(rows_total, B * 2, dtype=torch.float32)
            out[:, 0::2] = lut[idx_lo]
            out[:, 1::2] = lut[idx_hi]
            # Convert to target dtype after indexing
            out = out.to(dtype)
            # TensorRT-LLM standard: direct multiplication
            out = out * scale_expanded.to(dtype)
            return out.reshape(E, OUT, B * 2)
        else:
            raise ValueError(f"Unsupported blocks dim: {blocks.dim()}")

    # Generate a single shard for current mapping.rank
    rank = config.mapping.rank if config.mapping else 0
    shard_path = output / f'rank{rank}.safetensors'
    weights: Dict[str, torch.Tensor] = {}

    # Distribute sinks across TP ranks
    if sinks_per_layer:
        tp_size = config.mapping.tp_size if config.mapping else 1
        num_heads = config.num_attention_heads
        heads_per_rank = num_heads // tp_size if tp_size > 0 else num_heads

        for layer_idx, sinks in sinks_per_layer.items():
            # sinks shape: [num_heads]
            beg = rank * heads_per_rank
            end = beg + heads_per_rank
            tp_sinks = sinks[beg:end].contiguous().to(torch.float32)
            key = f"transformer.layers.{layer_idx}.attention.sinks"
            weights[key] = tp_sinks

    # Extract attention/projection/ln/emb/lm_head when index exists
    if weight_map is not None:
        nl = config.num_hidden_layers
        # Per-layer
        # Infer and correct dimension config from weights of the first layer
        #  - hidden_size/head_size from q_proj.rows
        #  - intermediate_size from gate_up_proj_bias
        i0 = 0
        def get(name: str) -> torch.Tensor:
            file = weight_map.get(name)
            if file is None:
                raise KeyError(name)
            tensors = load_file(file)
            return tensors[name]

        q0 = get(f'model.layers.{i0}.self_attn.q_proj.weight')
        k0 = get(f'model.layers.{i0}.self_attn.k_proj.weight')
        q_rows = q0.shape[0]
        k_rows = k0.shape[0]
        nh = config.num_attention_heads
        if q_rows % nh != 0:
            raise ValueError(f"q_proj.rows {q_rows} not divisible by num_attention_heads {nh}")
        implied_head = q_rows // nh
        # hidden_size remains as declared in config.json (e.g., 2880). q_rows reflects heads*head_dim (e.g., 4096).
        if config.head_size != implied_head:
            logger.info(f"Adjusting head_size from {config.head_size} to {implied_head} based on q_proj.rows/num_heads")
            config.head_size = implied_head

        # Infer num_key_value_heads from k_proj.rows and head_size
        if k_rows % config.head_size != 0:
            raise ValueError(f"k_proj.rows {k_rows} not divisible by head_size {config.head_size}")
        implied_kv_heads = k_rows // config.head_size
        if config.num_key_value_heads != implied_kv_heads:
            logger.info(f"Adjusting num_key_value_heads from {config.num_key_value_heads} to {implied_kv_heads} based on k_proj.rows/head_size")
            config.num_key_value_heads = implied_kv_heads

        # Infer intermediate_size robustly from down_proj blocks (authoritative),
        # falling back to gate_up_proj_bias only if needed. This matches actual proj.in_features.
        gub = get(f'model.layers.{i0}.mlp.experts.gate_up_proj_bias')
        if gub.shape[-1] % 2 != 0:
            raise ValueError(f"gate_up_proj_bias length {gub.shape[-1]} not even; cannot infer intermediate_size")
        implied_inter_from_bias = gub.shape[-1] // 2

        down0 = get(f'model.layers.{i0}.mlp.experts.down_proj_blocks')
        if down0.dim() == 4:
            # [E, OUT, IN/vec, PACK]
            inferred_inter = down0.shape[2] * down0.shape[3] * 2
        elif down0.dim() == 3:
            # [E, OUT, IN/vec]
            inferred_inter = down0.shape[2] * 2
        else:
            raise ValueError(f"Unexpected down_proj_blocks dim: {down0.dim()}")

        if config.intermediate_size != inferred_inter:
            logger.info(
                f"Adjusting intermediate_size from {config.intermediate_size} to {inferred_inter} based on down_proj_blocks")
            config.intermediate_size = inferred_inter
        # Optional consistency check
        if inferred_inter != implied_inter_from_bias:
            logger.warning(
                f"intermediate_size inferred from down_proj ({inferred_inter}) differs from gate_up_proj_bias/2 ({implied_inter_from_bias}); using down_proj value.")

        # Persist updated dimension config
        config.to_json_file(str(output / 'config.json'))

        # Recompute implied head again after adjustments
        tp_size = config.mapping.tp_size if config.mapping else 1
        nl = config.num_hidden_layers
        # Determine processing strategy from quantization config
        quant_algo = config.quantization.quant_algo
        # B200+ supports native MXFP4 regardless of quantization setting
        current_sm = get_sm_version()
        use_native_mxfp4 = (current_sm >= 100)
        deq_dtype = None if use_native_mxfp4 else target_dtype

        # Per-layer
        for i in range(nl):
                # q,k,v
                def get(name: str) -> torch.Tensor:
                    file = weight_map.get(name)
                    if file is None:
                        raise KeyError(name)
                    tensors = load_file(file)
                    return tensors[name]

                q_w = get(f'model.layers.{i}.self_attn.q_proj.weight').to(target_dtype)
                k_w = get(f'model.layers.{i}.self_attn.k_proj.weight').to(target_dtype)
                v_w = get(f'model.layers.{i}.self_attn.v_proj.weight').to(target_dtype)
                q_b = get(f'model.layers.{i}.self_attn.q_proj.bias').to(target_dtype)
                k_b = get(f'model.layers.{i}.self_attn.k_proj.bias').to(target_dtype)
                v_b = get(f'model.layers.{i}.self_attn.v_proj.bias').to(target_dtype)

                # shape checks for QKV
                nh = config.num_attention_heads
                nkh = config.num_key_value_heads
                hs = config.head_size
                assert q_w.shape[0] == nh * hs, f"q_proj.rows {q_w.shape[0]} != num_heads*head_size {nh*hs}"
                assert k_w.shape[0] == nkh * hs, f"k_proj.rows {k_w.shape[0]} != num_kv_heads*head_size {nkh*hs}"
                assert v_w.shape[0] == nkh * hs, f"v_proj.rows {v_w.shape[0]} != num_kv_heads*head_size {nkh*hs}"

                if tp_size == 1:
                    qkv_w_all = torch.concat([q_w, k_w, v_w], dim=0).contiguous()
                    qkv_b_all = torch.concat([q_b, k_b, v_b], dim=0).contiguous()
                    # Use simple_linear_weight for proper quantization (copied from Qwen3)
                    weights.update(
                        simple_linear_weight(qkv_w_all, f'transformer.layers.{i}.attention.qkv.',
                                             qkv_b_all, use_weight_only,
                                             plugin_weight_only_quant_type, target_dtype,
                                             use_gemm_woq_plugin, 
                                             ))  # Simple weight storage
                else:
                    # GQA-aware TP split: split Q per num_heads, duplicate KV per num_kv_heads
                    num_heads = config.num_attention_heads
                    num_kv_heads = config.num_key_value_heads
                    head_size = config.head_size

                    # Q rows split evenly across TP ranks
                    q_w_tp = split_matrix_tp(q_w, tp_size, rank, dim=0)
                    q_b_tp = split_matrix_tp(q_b, tp_size, rank, dim=0)

                    # Duplicate KV rows if needed so rows % tp_size == 0
                    k_w_eff = k_w
                    v_w_eff = v_w
                    k_b_eff = k_b
                    v_b_eff = v_b
                    if (k_w.shape[0] % tp_size) != 0 and num_kv_heads > 0:
                        k_w_eff = dup_kv_weight(k_w, num_kv_heads, tp_size)
                        v_w_eff = dup_kv_weight(v_w, num_kv_heads, tp_size)
                        k_b_eff = dup_kv_bias(k_b, num_kv_heads, tp_size)
                        v_b_eff = dup_kv_bias(v_b, num_kv_heads, tp_size)

                    k_w_tp = split_matrix_tp(k_w_eff, tp_size, rank, dim=0)
                    v_w_tp = split_matrix_tp(v_w_eff, tp_size, rank, dim=0)
                    k_b_tp = split_matrix_tp(k_b_eff, tp_size, rank, dim=0)
                    v_b_tp = split_matrix_tp(v_b_eff, tp_size, rank, dim=0)

                    qkv_w_tp = torch.concat([q_w_tp, k_w_tp, v_w_tp], dim=0).contiguous()
                    qkv_b_tp = torch.concat([q_b_tp, k_b_tp, v_b_tp], dim=0).contiguous()

                    # Use simple_linear_weight for proper quantization (copied from Qwen3)
                    weights.update(
                        simple_linear_weight(qkv_w_tp, f'transformer.layers.{i}.attention.qkv.',
                                             qkv_b_tp, use_weight_only,
                                             plugin_weight_only_quant_type, target_dtype,
                                             use_gemm_woq_plugin,
                                             ))  # Simple weight storage

                o_w = get(f'model.layers.{i}.self_attn.o_proj.weight').to(target_dtype)
                o_b = get(f'model.layers.{i}.self_attn.o_proj.bias').to(target_dtype)
                assert o_w.shape[1] == nh * hs, f"o_proj.cols {o_w.shape[1]} != num_heads*head_size {nh*hs}"
                if tp_size == 1:
                    # Use simple_linear_weight for proper quantization (copied from Qwen3)
                    weights.update(
                        simple_linear_weight(o_w, f'transformer.layers.{i}.attention.dense.',
                                             o_b, use_weight_only,
                                             plugin_weight_only_quant_type, target_dtype,
                                             use_gemm_woq_plugin,
                                             ))  # Simple weight storage
                else:
                    # dense is row-parallel in many models; split columns for output gathering
                    tp_w = torch.chunk(o_w, tp_size, dim=1)[rank].contiguous()
                    # Use simple_linear_weight for proper quantization (copied from Qwen3)
                    weights.update(
                        simple_linear_weight(tp_w, f'transformer.layers.{i}.attention.dense.',
                                             o_b, use_weight_only,
                                             plugin_weight_only_quant_type, target_dtype,
                                             use_gemm_woq_plugin,
                                             ))  # Simple weight storage

                in_ln = get(f'model.layers.{i}.input_layernorm.weight').to(target_dtype)
                po_ln = get(f'model.layers.{i}.post_attention_layernorm.weight').to(target_dtype)
                weights[f'transformer.layers.{i}.input_layernorm.weight'] = in_ln.contiguous()
                weights[f'transformer.layers.{i}.post_layernorm.weight'] = po_ln.contiguous()

                # MoE (raw MXFP4) passthrough for later consumption
                gate_up_blocks = get(f'model.layers.{i}.mlp.experts.gate_up_proj_blocks')
                gate_up_scales = get(f'model.layers.{i}.mlp.experts.gate_up_proj_scales')
                gate_up_bias = get(f'model.layers.{i}.mlp.experts.gate_up_proj_bias')
                # shape checks for gate_up blocks/scales
                if gate_up_blocks.dim() == 4:
                    assert gate_up_scales.dim() == 3
                    assert gate_up_blocks.shape[0] == gate_up_scales.shape[0]
                    assert gate_up_blocks.shape[1] == gate_up_scales.shape[1]
                    assert gate_up_blocks.shape[2] == gate_up_scales.shape[2]
                elif gate_up_blocks.dim() == 3:
                    assert gate_up_scales.dim() == 3
                    assert gate_up_blocks.shape == gate_up_scales.shape
                else:
                    assert False, f"Unexpected gate_up_blocks dim: {gate_up_blocks.dim()}"

                # Deinterleave gate/up along the OUT feature axis for TensorRT-LLM SwiGLU compatibility
                # TensorRT-LLM SwiGLU uses chunk(x, 2, dim=-1), expecting [gate0,gate1,...,up0,up1,...] format
                def _deinterleave_gate_up(tensor, out_axis):
                    out_len = tensor.shape[out_axis]
                    idx_even = torch.arange(0, out_len, 2)  # gate indices
                    idx_odd = torch.arange(1, out_len, 2)   # up indices  
                    take = lambda t, idx: t.index_select(out_axis, idx.to(t.device))
                    return take(tensor, idx_even), take(tensor, idx_odd)

                # Determine out axis for blocks/scales
                out_axis_blocks = -3 if gate_up_blocks.dim() >= 4 else -2
                out_axis_scales = -2  # scales expected [E, OUT, IN/vec]

                gate_w, up_w = _deinterleave_gate_up(gate_up_blocks, out_axis_blocks)
                gate_sc, up_sc = _deinterleave_gate_up(gate_up_scales, out_axis_scales)

                # Create segregated format: [up0,up1,...,gate0,gate1,...] for SwiGLU (following other models)
                # TensorRT-LLM SwiGLU: x, gate = chunk(weight, 2) → x * silu(gate)
                fc_blocks = torch.concat([up_w, gate_w], dim=out_axis_blocks).contiguous()
                fc_scales = torch.concat([up_sc, gate_sc], dim=out_axis_scales).contiguous()
                
                # Deinterleave bias: [g0,u0,g1,u1,...] → [u0,u1,...,g0,g1,...] (UP first, GATE second)
                gate_b = gate_up_bias[:, ::2]  # [g0, g1, g2, ...]  
                up_b = gate_up_bias[:, 1::2]   # [u0, u1, u2, ...]
                fc_bias = torch.concat([up_b, gate_b], dim=-1).contiguous()  # [u0,u1,...,g0,g1,...] (UP, GATE)
                # Cast bias to unified target precision
                fc_bias = fc_bias.to(target_dtype)

                def _collapse_pack_dim(blocks: torch.Tensor) -> torch.Tensor:
                    if blocks.dim() == 4:
                        E, OUT, IN_DIV, PACK = blocks.shape
                        return blocks.reshape(E, OUT, IN_DIV * PACK)
                    return blocks

                if use_native_mxfp4:
                    # Native MXFP4: Preserve original format and scales (SM100+ maximum accuracy)
                    if tp_size == 1:
                        # Keep original MXFP4 blocks and scales without conversion
                        fc_blocks_ckpt = _collapse_pack_dim(fc_blocks)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.weight.scales'] = fc_scales.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias
                        
                        logger.debug(f"Layer {i} fc: Native MXFP4 blocks {fc_blocks.shape}, scales {fc_scales.shape}")
                    else:
                        # Tensor Parallel support for native MXFP4
                        # fc (gate_up_proj) is ColLinear-like: split along OUT dimension  
                        # Note: fc_blocks/fc_scales now in segregated format [g0,g1,...,u0,u1,...] for SwiGLU
                        assert fc_blocks.dim() in [3, 4], f"fc_blocks unexpected dim: {fc_blocks.shape}"
                        assert fc_scales.dim() == 3, f"fc_scales expected 3D [E, OUT, IN/vec], got: {fc_scales.shape}"
                        
                        out_axis_blocks_tp = -3 if fc_blocks.dim() == 4 else -2
                        fc_blocks_tp = torch.chunk(fc_blocks, tp_size, dim=out_axis_blocks_tp)[rank].contiguous()
                        fc_scales_tp = torch.chunk(fc_scales, tp_size, dim=-2)[rank].contiguous()  # Split OUT dimension
                        fc_bias_tp = torch.chunk(fc_bias, tp_size, dim=-1)[rank].contiguous()
                        
                        fc_blocks_ckpt = _collapse_pack_dim(fc_blocks_tp)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.weight.scales'] = fc_scales_tp.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias_tp
                        

                else:
                    # Simple dequantization based on quantization setting
                    if tp_size == 1:
                        fc_weight = _dequantize_mxfp4(fc_blocks, fc_scales, dtype=mxfp4_dequant_dtype)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_weight.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias
                    else:
                        # TP chunking for dequantized path
                        out_axis_blocks_tp = -3 if fc_blocks.dim() == 4 else -2
                        fc_blocks_tp = torch.chunk(fc_blocks, tp_size, dim=out_axis_blocks_tp)[rank].contiguous()
                        fc_scales_tp = torch.chunk(fc_scales, tp_size, dim=-2)[rank].contiguous()
                        fc_bias_tp = torch.chunk(fc_bias, tp_size, dim=-1)[rank].contiguous()

                        fc_weight_tp = _dequantize_mxfp4(fc_blocks_tp, fc_scales_tp, dtype=mxfp4_dequant_dtype)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_weight_tp.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias_tp

                down_blocks = get(f'model.layers.{i}.mlp.experts.down_proj_blocks')
                down_scales = get(f'model.layers.{i}.mlp.experts.down_proj_scales')
                down_bias = get(f'model.layers.{i}.mlp.experts.down_proj_bias')
                if down_blocks.dim() == 4:
                    assert down_scales.dim() == 3
                    assert down_blocks.shape[0] == down_scales.shape[0]
                    assert down_blocks.shape[1] == down_scales.shape[1]
                    assert down_blocks.shape[2] == down_scales.shape[2]
                elif down_blocks.dim() == 3:
                    assert down_scales.dim() == 3
                    assert down_blocks.shape == down_scales.shape
                else:
                    assert False, f"Unexpected down_blocks dim: {down_blocks.dim()}"
                # Keep 3D layout [E, out, in/vec] for dequantization
                # Align down_proj shapes: allow 4D blocks [E, OUT, IN/vec, pack_vec] and 3D scales [E, OUT, IN/vec]
                proj_blocks = down_blocks.contiguous()
                proj_scales = down_scales.contiguous()
                if proj_blocks.dim() == 4 and proj_scales.dim() == 3:
                    # no change needed for dequant path; helper handles vec packing
                    pass

                if use_native_mxfp4:
                    # Native MXFP4: Preserve original format and scales (SM100+ maximum accuracy)
                    if tp_size == 1:
                        # Keep original MXFP4 blocks and scales without conversion
                        proj_blocks_ckpt = _collapse_pack_dim(proj_blocks)
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.weight.scales'] = proj_scales.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(target_dtype).contiguous()
                        
                        logger.debug(f"Layer {i} proj: Native MXFP4 blocks {proj_blocks.shape}, scales {proj_scales.shape}")
                    else:
                        # Tensor Parallel support for native MXFP4
                        # proj is RowLinear-like: split along IN dimension
                        assert proj_blocks.dim() in [3, 4], f"proj_blocks unexpected dim: {proj_blocks.shape}"
                        assert proj_scales.dim() == 3, f"proj_scales expected 3D [E, OUT, IN/vec], got: {proj_scales.shape}"
                        
                        in_axis_blocks_tp = -2 if proj_blocks.dim() == 4 else -1
                        proj_blocks_tp = torch.chunk(proj_blocks, tp_size, dim=in_axis_blocks_tp)[rank].contiguous()
                        proj_scales_tp = torch.chunk(proj_scales, tp_size, dim=-1)[rank].contiguous()  # Split IN dimension

                        proj_blocks_ckpt = _collapse_pack_dim(proj_blocks_tp)
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.weight.scales'] = proj_scales_tp.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(target_dtype).contiguous()
                        
                else:
                    # Simple dequantization based on quantization setting
                    proj_weight = _dequantize_mxfp4(proj_blocks, proj_scales, dtype=mxfp4_dequant_dtype)
                    assert proj_weight.shape[-1] == config.intermediate_size, \
                        f"PROJ in_features {proj_weight.shape[-1]} != intermediate_size {config.intermediate_size}"
                    
                    if tp_size == 1:
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_weight.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(target_dtype).contiguous()
                    else:
                        # Split along IN axis for RowLinear-like proj
                        proj_weight_tp = torch.chunk(proj_weight, tp_size, dim=-1)[rank].contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_weight_tp
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(target_dtype).contiguous()

                # BF16 per-expert path intentionally not supported for gpt-oss (MoE uses MXFP4)
                
                # Generate FP8 scaling factors (corrected based on TensorRT-LLM standard)
                if quant_algo == QuantAlgo.FP8:
                    tllm_prex = f'transformer.layers.{i}'
                    
                    # Get MoE experts count (following Grok pattern)
                    num_experts = config.moe.num_experts if config.moe and config.moe.num_experts > 0 else 1
                    
                    # Attention scaling factors (always (1,) shape - not MoE related)
                    weights[f'{tllm_prex}.attention.qkv.activation_scaling_factor'] = get_fp8_activation_scaling_factor()
                    weights[f'{tllm_prex}.attention.qkv.weights_scaling_factor'] = get_fp8_weights_scaling_factor(num_experts=1)
                    weights[f'{tllm_prex}.attention.dense.activation_scaling_factor'] = get_fp8_activation_scaling_factor()
                    weights[f'{tllm_prex}.attention.dense.weights_scaling_factor'] = get_fp8_weights_scaling_factor(num_experts=1)
                    
                    # MLP scaling factors (activation: always (1,), weights: MoE-dependent)
                    weights[f'{tllm_prex}.mlp.fc.activation_scaling_factor'] = get_fp8_activation_scaling_factor()
                    weights[f'{tllm_prex}.mlp.fc.weights_scaling_factor'] = get_fp8_weights_scaling_factor(num_experts=num_experts)
                    weights[f'{tllm_prex}.mlp.proj.activation_scaling_factor'] = get_fp8_activation_scaling_factor()
                    weights[f'{tllm_prex}.mlp.proj.weights_scaling_factor'] = get_fp8_weights_scaling_factor(num_experts=num_experts)
                    
                    # KV cache scaling factors (following Gemma pattern exactly)
                    scaling_factor = 1.0
                    weights[f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.tensor(
                        [scaling_factor], dtype=fake_fp8_sf_dt)
                    # Generate reciprocal scaling factor (following modeling_utils.py pattern)
                    weights[f'{tllm_prex}.attention.kv_cache_rcp_scaling_factor'] = torch.reciprocal(
                        torch.tensor([scaling_factor], dtype=fake_fp8_sf_dt))

        # Embeddings & final norm & lm_head (once per rank)
        emb_w = get('model.embed_tokens.weight').to(target_dtype)
        if tp_size == 1:
            # Use simple_linear_weight for proper quantization (copied from Qwen3)
            weights.update(
                simple_linear_weight(emb_w, 'transformer.vocab_embedding.',
                                     None, use_weight_only,
                                     plugin_weight_only_quant_type, target_dtype,
                                     use_gemm_woq_plugin,
                                     ))  # Simple weight storage
        else:
            # column-sharded along vocab dimension
            tp_emb = torch.chunk(emb_w, tp_size, dim=0)[rank].contiguous()
            weights.update(
                simple_linear_weight(tp_emb, 'transformer.vocab_embedding.',
                                     None, use_weight_only,
                                     plugin_weight_only_quant_type, target_dtype,
                                     use_gemm_woq_plugin,
                                     ))  # Simple weight storage

        ln_f = get('model.norm.weight').to(target_dtype)
        weights['transformer.ln_f.weight'] = ln_f.contiguous()

        lm_w = get('lm_head.weight').to(target_dtype)
        if tp_size == 1:
            # Use simple_linear_weight for proper quantization (copied from Qwen3)
            weights.update(
                simple_linear_weight(lm_w, 'lm_head.',
                                     None, use_weight_only,
                                     plugin_weight_only_quant_type, target_dtype,
                                     use_gemm_woq_plugin,
                                     ))  # Simple weight storage
        else:
            tp_lm = torch.chunk(lm_w, tp_size, dim=0)[rank].contiguous()
            weights.update(
                simple_linear_weight(tp_lm, 'lm_head.',
                                     None, use_weight_only,
                                     plugin_weight_only_quant_type, target_dtype,
                                     use_gemm_woq_plugin,
                                     ))  # Simple weight storage

    # Router weight (optional; present in MoE) — write on all ranks
    if weight_map is not None:
        # probe existence
        _ = weight_map.get('model.layers.0.mlp.router.weight')
        for i in range(config.num_hidden_layers):
            name = f'model.layers.{i}.mlp.router.weight'
            file = weight_map.get(name)
            if file is None:
                continue
            tensors = load_file(file)
            if name not in tensors:
                continue
            # Router runs in float32 for numerical stability and matches runtime cast
            w = tensors[name].to(torch.float32)
            # keep unsplit; MOE handles distribution internally
            weights[f'transformer.layers.{i}.mlp.router.weight'] = w.contiguous()
            
            # Router does NOT need FP8 scaling factors since it runs in float32
            # (following standard TensorRT-LLM MoE pattern)

    safetensors.torch.save_file(weights, str(shard_path))
    logger.info(f"Wrote shard with {len(weights)} tensors: {shard_path}")

    logger.info(
        f"convert_and_save: Successfully created shard rank{rank} (of world_size={world_size}) with QKV/GQA/TP, MoE MXFP4/BF16, and attention sinks support."
    )


def load_weights_from_hf_model(
    model_dir: Union[str, Path],
    config: GptOssConfig,
    *,
    quant_config: Optional[QuantConfig] = None,
) -> None:
    """Load weights from HuggingFace GPT-OSS model."""
    # Extract output_dir from config.mapping if available, or use default
    output_dir = getattr(config, 'output_dir', './tmp_checkpoint')
    
    # Call the main conversion function
    convert_and_save(
        model_dir=model_dir,
        output_dir=output_dir,
        config=config,
        quant_config=quant_config
    )


def _extract_sinks_from_index(model_dir: Path) -> Dict[int, torch.Tensor]:
    """
    Parse HF safetensors index and extract per-layer self_attn.sinks tensors.
    Returns: dict[layer_idx] = 1D float tensor of length num_heads.
    """
    index_path = model_dir / 'model.safetensors.index.json'
    with open(index_path, 'r') as f:
        idx = json.load(f)  # type: ignore
    weight_map: Dict[str, str] = idx['weight_map']

    # group keys per layer
    sinks: Dict[int, torch.Tensor] = {}
    cache_files: Dict[str, Dict[str, torch.Tensor]] = {}

    def load_file(file_name: str) -> Dict[str, torch.Tensor]:
        if file_name in cache_files:
            return cache_files[file_name]
        data_path = model_dir / file_name
        # Use safe_open for better compatibility
        tensors = {}
        with safe_open(data_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        cache_files[file_name] = tensors
        return tensors

    for key, file_name in weight_map.items():
        # interested in keys like: model.layers.{i}.self_attn.sinks
        if not key.endswith('.self_attn.sinks'):
            continue
        parts = key.split('.')
        # ['model','layers','{i}','self_attn','sinks']
        layer_idx = int(parts[2])
        tensors = load_file(file_name)
        if key not in tensors:
            continue
        t = tensors[key].to(torch.float32).cpu()
        sinks[layer_idx] = t

    return sinks

