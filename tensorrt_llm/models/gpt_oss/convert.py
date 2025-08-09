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

    # Create minimal rank shards to allow plumbing tests
    world_size = config.mapping.world_size if config.mapping else 1
    for rank in range(world_size):
        shard_path = output / f'rank{rank}.safetensors'
        if shard_path.exists():
            continue
        safetensors.torch.save_file({}, str(shard_path))
        logger.info(f"Created placeholder shard: {shard_path}")

    logger.warning(
        "convert_and_save: Placeholder shards created. Full weight conversion will be implemented next (QKV/GQA/TP, MoE MXFP4, sinks)."
    )


