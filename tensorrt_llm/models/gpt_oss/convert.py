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

