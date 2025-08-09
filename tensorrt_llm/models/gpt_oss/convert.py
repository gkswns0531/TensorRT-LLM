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
    assert tp_size == 1, "Initial converter supports TP=1 only; implement TP split in a later step."

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
        if weight_map is not None and rank == 0:
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
                    # Concat along out_dim for weights; biases along dim 0
                    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
                    qkv_b = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
                    weights[f'transformer.layers.{i}.attention.qkv.weight'] = qkv_w
                    weights[f'transformer.layers.{i}.attention.qkv.bias'] = qkv_b
                except Exception as e:
                    logger.warning(f"layer {i}: skipping qkv ({e})")

                try:
                    o_w = get(f'model.layers.{i}.self_attn.o_proj.weight')
                    o_b = get(f'model.layers.{i}.self_attn.o_proj.bias')
                    weights[f'transformer.layers.{i}.attention.dense.weight'] = o_w.contiguous()
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
                    weights[f'transformer.layers.{i}.mlp.experts.gate_up_proj.blocks'] = gate_up_blocks.contiguous()
                    weights[f'transformer.layers.{i}.mlp.experts.gate_up_proj.scales'] = gate_up_scales.contiguous()
                    weights[f'transformer.layers.{i}.mlp.experts.gate_up_proj.bias'] = gate_up_bias.contiguous()
                except Exception as e:
                    logger.info(f"layer {i}: gate_up_proj not found ({e})")

                try:
                    down_blocks = get(f'model.layers.{i}.mlp.experts.down_proj_blocks')
                    down_scales = get(f'model.layers.{i}.mlp.experts.down_proj_scales')
                    down_bias = get(f'model.layers.{i}.mlp.experts.down_proj_bias')
                    weights[f'transformer.layers.{i}.mlp.experts.down_proj.blocks'] = down_blocks.contiguous()
                    weights[f'transformer.layers.{i}.mlp.experts.down_proj.scales'] = down_scales.contiguous()
                    weights[f'transformer.layers.{i}.mlp.experts.down_proj.bias'] = down_bias.contiguous()
                except Exception as e:
                    logger.info(f"layer {i}: down_proj not found ({e})")

            # Embeddings & final norm & lm_head
            try:
                emb_w = get('model.embed_tokens.weight')
                weights['transformer.vocab_embedding.weight'] = emb_w.contiguous()
            except Exception as e:
                logger.warning(f"embed skip: {e}")

            try:
                ln_f = get('model.norm.weight')
                weights['transformer.ln_f.weight'] = ln_f.contiguous()
            except Exception as e:
                logger.warning(f"ln_f skip: {e}")

            try:
                lm_w = get('lm_head.weight')
                weights['lm_head.weight'] = lm_w.contiguous()
            except Exception as e:
                logger.warning(f"lm_head skip: {e}")

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

