"""
Weights conversion utilities for GPT-OSS to TensorRT-LLM checkpoint format.

This module provides scaffolding for:
 - Loading HF or original checkpoints
 - Extracting/reshaping Q/K/V/O projections with GQA and TP
 - Preparing MoE (gate_up_proj/down_proj) tensors (BF16/MXFP4)
 - Attention sinks extraction per layer

Note: This is an initial scaffold. Full MXFP4 and MoE routing details
will be implemented incrementally, following existing model patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import safetensors
import torch

from ...logger import logger
from ...mapping import Mapping
from ..modeling_utils import QuantConfig
from ..convert_utils import split_matrix_tp, dup_kv_weight, dup_kv_bias
from .config import GptOssConfig


@dataclass
class ConvertContext:
    config: GptOssConfig
    mapping: Mapping
    quant_config: QuantConfig
    model_dir: Path
    use_hf: bool


def _detect_hf_or_original(model_dir: Union[str, Path]) -> Tuple[bool, Path]:
    p = Path(model_dir)
    orig = p / 'original' / 'model.safetensors'
    return (not orig.exists(), p)


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
    """
    Convert the weights into TensorRT-LLM checkpoint shards (rank*.safetensors).

    This initial version writes config only, and stubs empty rank0 weight files
    to bootstrap the pipeline. Weight population will be added incrementally.
    """
    use_hf, p_model = _detect_hf_or_original(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Save config.json (already handled by caller typically)
    config.to_json_file(str(output / 'config.json'))

    # Prepare minimal shards and try to import attention sinks
    world_size = config.mapping.world_size if config.mapping else 1
    sinks_per_layer = {}
    try:
        sinks_per_layer = _extract_sinks_from_index(p_model)
        logger.info(f"Loaded sinks for {len(sinks_per_layer)} layers")
    except Exception as e:
        logger.warning(f"Failed to load sinks (optional): {e}")

    # Load HF weight_map for QKV/Norm/Embeddings/lm_head
    weight_map = None
    try:
        with open(p_model / 'model.safetensors.index.json', 'r') as f:
            idx = json.load(f)
        weight_map = idx['weight_map']
    except Exception as e:
        logger.warning(f"Index not found or unreadable, skipping weights: {e}")

    cache_files: Dict[str, Dict[str, torch.Tensor]] = {}

    def load_file(file_name: str) -> Dict[str, torch.Tensor]:
        if file_name in cache_files:
            return cache_files[file_name]
        data_path = p_model / file_name
        tensors = safetensors.torch.load_file(str(data_path))
        cache_files[file_name] = tensors
        return tensors

    tp_size = config.mapping.tp_size if config.mapping else 1

    for rank in range(world_size):
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
            for i in range(nl):
                # q,k,v
                def get(name: str) -> torch.Tensor:
                    file = weight_map.get(name)
                    if file is None:
                        raise KeyError(name)
                    tensors = load_file(file)
                    return tensors[name]

                try:
                    q_w = get(f'model.layers.{i}.self_attn.q_proj.weight')
                    k_w = get(f'model.layers.{i}.self_attn.k_proj.weight')
                    v_w = get(f'model.layers.{i}.self_attn.v_proj.weight')
                    q_b = get(f'model.layers.{i}.self_attn.q_proj.bias')
                    k_b = get(f'model.layers.{i}.self_attn.k_proj.bias')
                    v_b = get(f'model.layers.{i}.self_attn.v_proj.bias')

                    if tp_size == 1:
                        qkv_w_all = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
                        qkv_b_all = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
                        weights[f'transformer.layers.{i}.attention.qkv.weight'] = qkv_w_all
                        weights[f'transformer.layers.{i}.attention.qkv.bias'] = qkv_b_all
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

                        qkv_w_tp = torch.cat([q_w_tp, k_w_tp, v_w_tp], dim=0).contiguous()
                        qkv_b_tp = torch.cat([q_b_tp, k_b_tp, v_b_tp], dim=0).contiguous()

                        weights[f'transformer.layers.{i}.attention.qkv.weight'] = qkv_w_tp
                        weights[f'transformer.layers.{i}.attention.qkv.bias'] = qkv_b_tp
                except Exception as e:
                    logger.warning(f"layer {i}: skipping qkv ({e})")

                try:
                    o_w = get(f'model.layers.{i}.self_attn.o_proj.weight')
                    o_b = get(f'model.layers.{i}.self_attn.o_proj.bias')
                    if tp_size == 1:
                        weights[f'transformer.layers.{i}.attention.dense.weight'] = o_w.contiguous()
                        weights[f'transformer.layers.{i}.attention.dense.bias'] = o_b.contiguous()
                    else:
                        # dense is row-parallel in many models; split columns for output gathering
                        tp_w = torch.chunk(o_w, tp_size, dim=1)[rank].contiguous()
                        weights[f'transformer.layers.{i}.attention.dense.weight'] = tp_w
                        # Write bias on all ranks; non-zero ranks will be zeroed in preprocess
                        weights[f'transformer.layers.{i}.attention.dense.bias'] = o_b.contiguous()
                except Exception as e:
                    logger.warning(f"layer {i}: skipping o_proj ({e})")

                try:
                    in_ln = get(f'model.layers.{i}.input_layernorm.weight')
                    po_ln = get(f'model.layers.{i}.post_attention_layernorm.weight')
                    weights[f'transformer.layers.{i}.input_layernorm.weight'] = in_ln.contiguous()
                    weights[f'transformer.layers.{i}.post_layernorm.weight'] = po_ln.contiguous()
                except Exception as e:
                    logger.warning(f"layer {i}: skipping norms ({e})")

                # MoE (raw MXFP4) passthrough for later consumption
                try:
                    gate_up_blocks = get(f'model.layers.{i}.mlp.experts.gate_up_proj_blocks')
                    gate_up_scales = get(f'model.layers.{i}.mlp.experts.gate_up_proj_scales')
                    gate_up_bias = get(f'model.layers.{i}.mlp.experts.gate_up_proj_bias')
                    # Deinterleave gate/up along expert channel axis then concatenate
                    gub_flat = gate_up_blocks.flatten(-2, -1)
                    gub_sc_flat = gate_up_scales.flatten(-2, -1)
                    gate_w = gub_flat[:, ::2, :]
                    up_w = gub_flat[:, 1::2, :]
                    gate_sc = gub_sc_flat[:, ::2, :]
                    up_sc = gub_sc_flat[:, 1::2, :]
                    # Concatenate gate, up along channel axis
                    fc_blocks = torch.cat([gate_w, up_w], dim=-2).contiguous()
                    fc_scales = torch.cat([gate_sc, up_sc], dim=-2).contiguous()
                    # Bias: interleaved pairs gate/up across last dim
                    gate_b = gate_up_bias[:, ::2]
                    up_b = gate_up_bias[:, 1::2]
                    fc_bias = torch.cat([gate_b, up_b], dim=-1).contiguous()

                    if tp_size == 1:
                        weights[f'transformer.layers.{i}.mlp.fc.blocks'] = fc_blocks
                        weights[f'transformer.layers.{i}.mlp.fc.scales'] = fc_scales
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias
                        # NVFP4 expected auxiliary scales
                        try:
                            num_experts = fc_scales.shape[0]
                            wbsf = fc_scales.to(torch.uint8).contiguous()
                            weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor'] = wbsf
                            inter = torch.ops.trtllm.block_scale_interleave(wbsf)
                            weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor_interleaved'] = inter
                            weights[f'transformer.layers.{i}.mlp.fc.activation_global_scaling_factor'] = torch.ones((1, ), dtype=torch.float32)
                            weights[f'transformer.layers.{i}.mlp.fc.alpha'] = torch.ones((num_experts, ), dtype=torch.float32)
                        except Exception:
                            pass
                    else:
                        # Split along output-channel axis (-2) for blocks/scales and along last dim for bias
                        fc_blocks_tp = torch.chunk(fc_blocks, tp_size, dim=-2)[rank].contiguous()
                        fc_scales_tp = torch.chunk(fc_scales, tp_size, dim=-2)[rank].contiguous()
                        fc_bias_tp = torch.chunk(fc_bias, tp_size, dim=-1)[rank].contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.blocks'] = fc_blocks_tp
                        weights[f'transformer.layers.{i}.mlp.fc.scales'] = fc_scales_tp
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias_tp
                        try:
                            num_experts = fc_scales_tp.shape[0]
                            wbsf = fc_scales_tp.to(torch.uint8).contiguous()
                            weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor'] = wbsf
                            inter = torch.ops.trtllm.block_scale_interleave(wbsf)
                            weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor_interleaved'] = inter
                            weights[f'transformer.layers.{i}.mlp.fc.activation_global_scaling_factor'] = torch.ones((1, ), dtype=torch.float32)
                            weights[f'transformer.layers.{i}.mlp.fc.alpha'] = torch.ones((num_experts, ), dtype=torch.float32)
                        except Exception:
                            pass
                except Exception as e:
                    logger.info(f"layer {i}: gate_up_proj not found ({e})")

                try:
                    down_blocks = get(f'model.layers.{i}.mlp.experts.down_proj_blocks')
                    down_scales = get(f'model.layers.{i}.mlp.experts.down_proj_scales')
                    down_bias = get(f'model.layers.{i}.mlp.experts.down_proj_bias')
                    proj_blocks = down_blocks.flatten(-2, -1).contiguous()
                    proj_scales = down_scales.flatten(-2, -1).contiguous()

                    if tp_size == 1:
                        weights[f'transformer.layers.{i}.mlp.proj.blocks'] = proj_blocks
                        weights[f'transformer.layers.{i}.mlp.proj.scales'] = proj_scales
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.contiguous()
                        try:
                            num_experts = proj_scales.shape[0]
                            wbsf = proj_scales.to(torch.uint8).contiguous()
                            weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor'] = wbsf
                            inter = torch.ops.trtllm.block_scale_interleave(wbsf)
                            weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor_interleaved'] = inter
                            weights[f'transformer.layers.{i}.mlp.proj.activation_global_scaling_factor'] = torch.ones((1, ), dtype=torch.float32)
                            weights[f'transformer.layers.{i}.mlp.proj.alpha'] = torch.ones((num_experts, ), dtype=torch.float32)
                        except Exception:
                            pass
                    else:
                        # Split along input axis (last dim) for blocks/scales
                        proj_blocks_tp = torch.chunk(proj_blocks, tp_size, dim=-1)[rank].contiguous()
                        proj_scales_tp = torch.chunk(proj_scales, tp_size, dim=-1)[rank].contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.blocks'] = proj_blocks_tp
                        weights[f'transformer.layers.{i}.mlp.proj.scales'] = proj_scales_tp
                        # Record full bias on all ranks; non-zero ranks will be zeroed in preprocess
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.contiguous()
                        try:
                            num_experts = proj_scales_tp.shape[0]
                            wbsf = proj_scales_tp.to(torch.uint8).contiguous()
                            weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor'] = wbsf
                            inter = torch.ops.trtllm.block_scale_interleave(wbsf)
                            weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor_interleaved'] = inter
                            weights[f'transformer.layers.{i}.mlp.proj.activation_global_scaling_factor'] = torch.ones((1, ), dtype=torch.float32)
                            weights[f'transformer.layers.{i}.mlp.proj.alpha'] = torch.ones((num_experts, ), dtype=torch.float32)
                        except Exception:
                            pass
                except Exception as e:
                    logger.info(f"layer {i}: down_proj not found ({e})")

                # Optional BF16 path: if float weights exist, emit fc/proj.weight for MOE
                try:
                    gu_w_f = get(f'model.layers.{i}.mlp.experts.gate_up_proj.weight')
                    dp_w_f = get(f'model.layers.{i}.mlp.experts.down_proj.weight')
                    # gate_up float layout: [E, 2*ffn, H] or [2*ffn, H] when shared across experts
                    # Keep as-is to feed MOE.fc (expects gated 2*ffn)
                    if gu_w_f.dim() == 3:
                        fc_w = gu_w_f
                    else:
                        # expand to [E, 2*ffn, H] with E=1
                        fc_w = gu_w_f.unsqueeze(0)

                    if dp_w_f.dim() == 3:
                        proj_w = dp_w_f
                    else:
                        proj_w = dp_w_f.unsqueeze(0)

                    # TP split
                    if tp_size == 1:
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_w.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_w.contiguous()
                    else:
                        # fc: split along output-channel axis (-2)
                        fc_w_tp = torch.chunk(fc_w, tp_size, dim=-2)[rank].contiguous()
                        # proj: split along input axis (-1)
                        proj_w_tp = torch.chunk(proj_w, tp_size, dim=-1)[rank].contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_w_tp
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_w_tp
                except Exception:
                    pass

            # Embeddings & final norm & lm_head
            try:
                emb_w = get('model.embed_tokens.weight')
                if tp_size == 1:
                    weights['transformer.vocab_embedding.weight'] = emb_w.contiguous()
                else:
                    # column-sharded along vocab dimension
                    tp_emb = torch.chunk(emb_w, tp_size, dim=0)[rank].contiguous()
                    weights['transformer.vocab_embedding.weight'] = tp_emb
            except Exception as e:
                logger.warning(f"embed skip: {e}")

            try:
                ln_f = get('model.norm.weight')
                weights['transformer.ln_f.weight'] = ln_f.contiguous()
            except Exception as e:
                logger.warning(f"ln_f skip: {e}")

            try:
                lm_w = get('lm_head.weight')
                if tp_size == 1:
                    weights['lm_head.weight'] = lm_w.contiguous()
                else:
                    tp_lm = torch.chunk(lm_w, tp_size, dim=0)[rank].contiguous()
                    weights['lm_head.weight'] = tp_lm
            except Exception as e:
                logger.warning(f"lm_head skip: {e}")

        # Router weight (optional; present in MoE) — write on all ranks
        if weight_map is not None:
            try:
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
                    w = tensors[name]
                    # keep unsplit; MOE handles distribution internally
                    weights[f'transformer.layers.{i}.mlp.router.weight'] = w.contiguous()
            except Exception as e:
                logger.info(f"router skip: {e}")

        safetensors.torch.save_file(weights, str(shard_path))
        logger.info(f"Wrote shard with {len(weights)} tensors: {shard_path}")

    logger.warning(
        "convert_and_save: Placeholder shards created. Full weight conversion will be implemented next (QKV/GQA/TP, MoE MXFP4, sinks)."
    )


# --- Implementation guides (to be filled in next iterations) ---

def _extract_qkv_from_hf(hf_model, layer_idx: int, config: GptOssConfig) -> Dict[str, torch.Tensor]:
    """TODO: Read HF layer q,k,v weights + bias; return raw tensors prior to TP split."""
    raise NotImplementedError


def _tp_split_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mapping: Mapping, num_heads: int,
                  num_kv_heads: int, head_size: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """TODO: Perform TP split and KV duplication for GQA. Return weight/bias dicts per TP rank."""
    raise NotImplementedError


def _extract_moe_blocks_and_scales(hf_weights: Dict[str, torch.Tensor], config: GptOssConfig) -> Dict[str, torch.Tensor]:
    """TODO: Handle MXFP4 blocks+scales for gate_up_proj/down_proj, deinterleave gate/up and prepare per-expert tensors."""
    raise NotImplementedError


def _load_attention_sinks(hf_model_or_dir: Union[str, Path], num_layers: int, num_heads: int,
                          mapping: Mapping) -> Dict[int, torch.Tensor]:
    """TODO: Load per-layer sinks (float32 per head) and TP-slice them."""
    raise NotImplementedError


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
        tensors = safetensors.torch.load_file(str(data_path))
        cache_files[file_name] = tensors
        return tensors

    for key, file_name in weight_map.items():
        # interested in keys like: model.layers.{i}.self_attn.sinks
        if not key.endswith('.self_attn.sinks'):
            continue
        try:
            parts = key.split('.')
            # ['model','layers','{i}','self_attn','sinks']
            layer_idx = int(parts[2])
        except Exception:
            continue
        tensors = load_file(file_name)
        if key not in tensors:
            continue
        t = tensors[key].to(torch.float32).cpu()
        sinks[layer_idx] = t

    return sinks

