#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert Exaone 4.0 HuggingFace checkpoint to TensorRT-LLM format.

This script converts Exaone 4.0 models to TensorRT-LLM checkpoints that can be used
to build optimized TensorRT engines for inference.

Example usage:
    # Single GPU
    python convert_checkpoint.py \
        --model_dir ./hf_models/exaone-4.0-32b \
        --output_dir ./checkpoints/exaone-4.0-32b/fp16/1-gpu \
        --dtype float16

    # Tensor Parallel (4 GPUs)  
    python convert_checkpoint.py \
        --model_dir ./hf_models/exaone-4.0-32b \
        --output_dir ./checkpoints/exaone-4.0-32b/fp16/4-gpu \
        --dtype float16 \
        --tp_size 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tensorrt_llm.logger import logger  
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.exaone.config import Exaone4Config
from tensorrt_llm.models.exaone.model import Exaone4ForCausalLM
from tensorrt_llm.models.exaone.convert import load_exaone4_weights_from_hf_model
from tensorrt_llm.models.modeling_utils import save_checkpoint, save_config
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.quantization.mode import MODELOPT_FLOW_QUANTIZATIONS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Exaone 4.0 HuggingFace checkpoint to TensorRT-LLM format'
    )
    
    # Required arguments
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to the HuggingFace model directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for TensorRT-LLM checkpoint')
    
    # Model configuration
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model data type (default: bfloat16)')
    parser.add_argument('--logits_dtype', type=str, default='float32',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Logits data type (default: float32)')
    
    # Parallelization
    parser.add_argument('--tp_size', type=int, default=1,
                        help='Tensor parallelism size (default: 1)')
    parser.add_argument('--pp_size', type=int, default=1,
                        help='Pipeline parallelism size (default: 1)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of GPUs (tp_size * pp_size, default: 1)')
    
    # Quantization
    parser.add_argument('--use_weight_only', action='store_true',
                        help='Enable weight-only quantization')
    parser.add_argument('--weight_only_precision', type=str, default='int8',
                        choices=['int8', 'int4', 'int4_awq', 'int4_gptq'],
                        help='Weight-only quantization precision (default: int8)')
    parser.add_argument('--per_channel', action='store_true',
                        help='Use per-channel quantization (default: per-tensor)')
    parser.add_argument('--per_token', action='store_true',
                        help='Use per-token dynamic scaling for INT8-SQ')
    parser.add_argument('--int8_kv_cache', action='store_true',
                        help='Use INT8 quantization for KV cache')
    parser.add_argument('--fp8_kv_cache', action='store_true',
                        help='Use FP8 quantization for KV cache')
    
    # Model-specific options
    parser.add_argument('--use_parallel_embedding', action='store_true',
                        help='Use parallel embedding (needed when vocab_size % tp_size != 0)')
    parser.add_argument('--embedding_sharding_dim', type=int, default=0,
                        choices=[0, 1], help='Dimension to shard embedding (0: vocab, 1: hidden)')
    parser.add_argument('--share_embedding_table', action='store_true',
                        help='Share embedding table between encoder and decoder (saves memory)')
    
    # Advanced options
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes for conversion (default: 1)')
    parser.add_argument('--log_level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level (default: info)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check model directory exists
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    
    # Check world size consistency
    if args.world_size != args.tp_size * args.pp_size:
        logger.warning(
            f"world_size ({args.world_size}) != tp_size ({args.tp_size}) * pp_size ({args.pp_size}). "
            f"Setting world_size to {args.tp_size * args.pp_size}"
        )
        args.world_size = args.tp_size * args.pp_size
    
    # Validate quantization options
    if args.use_weight_only and args.weight_only_precision in ['int4_awq', 'int4_gptq']:
        logger.info(f"Using {args.weight_only_precision} quantization")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args


def setup_quantization(args, hf_config):
    """Setup quantization configuration."""
    quant_algo = None
    
    if args.use_weight_only:
        if args.weight_only_precision == 'int4_awq':
            quant_algo = QuantAlgo.W4A16_AWQ
        elif args.weight_only_precision == 'int4_gptq':
            quant_algo = QuantAlgo.W4A16_GPTQ
        elif args.weight_only_precision == 'int4':
            quant_algo = QuantAlgo.W4A16
        elif args.weight_only_precision == 'int8':
            quant_algo = QuantAlgo.W8A16
    
    kv_cache_quant_algo = None
    if args.int8_kv_cache:
        kv_cache_quant_algo = QuantAlgo.INT8
    elif args.fp8_kv_cache:
        kv_cache_quant_algo = QuantAlgo.FP8
    
    return quant_algo, kv_cache_quant_algo


def convert_checkpoint_worker(args, rank):
    """Worker function for checkpoint conversion."""
    logger.info(f"Converting checkpoint for rank {rank}")
    
    # Load HuggingFace config
    hf_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # Setup quantization
    quant_algo, kv_cache_quant_algo = setup_quantization(args, hf_config)
    
    # Create mapping for this rank
    mapping = Mapping(
        world_size=args.world_size,
        rank=rank,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )
    
    # Create TensorRT-LLM config
    config = Exaone4Config.from_hugging_face(
        hf_config_or_dir=hf_config,
        dtype=args.dtype,
        mapping=mapping,
    )
    
    # Set additional config options
    config.logits_dtype = args.logits_dtype
    config.use_parallel_embedding = args.use_parallel_embedding
    config.embedding_sharding_dim = args.embedding_sharding_dim
    
    # Apply quantization settings
    if quant_algo is not None:
        config.quant_mode = quant_algo
    if kv_cache_quant_algo is not None:
        config.kv_cache_quant_algo = kv_cache_quant_algo
    
    # Create model
    model = Exaone4ForCausalLM(config)
    
    # Load and convert weights
    weights = load_exaone4_weights_from_hf_model(args.model_dir, config, model)
    
    # Save checkpoint for this rank
    save_checkpoint(args.output_dir, config, model, weights)
    
    logger.info(f"Successfully converted checkpoint for rank {rank}")


def main():
    """Main conversion function."""
    print("="*50)
    print("Exaone 4.0 TensorRT-LLM Checkpoint Converter")
    print("="*50)
    
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    
    # Set up logging
    logger.set_level(args.log_level.upper())
    if args.verbose:
        logger.set_level("DEBUG")
    
    # Print configuration
    print(f"Input model: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data type: {args.dtype}")
    print(f"Tensor Parallelism: {args.tp_size}")
    print(f"Pipeline Parallelism: {args.pp_size}")
    if args.use_weight_only:
        print(f"Weight-only quantization: {args.weight_only_precision}")
    print()
    
    # Start conversion
    start_time = time.time()
    
    try:
        # Convert checkpoints for all ranks
        for rank in range(args.world_size):
            convert_checkpoint_worker(args, rank)
        
        # Save global config
        hf_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        mapping = Mapping(world_size=args.world_size, tp_size=args.tp_size, pp_size=args.pp_size)
        
        config = Exaone4Config.from_hugging_face(
            hf_config_or_dir=hf_config,
            dtype=args.dtype,
            mapping=mapping,
        )
        
        save_config(config, args.output_dir)
        
        print("="*50)
        print("✅ Conversion completed successfully!")
        print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
        print(f"📁 Checkpoint saved to: {args.output_dir}")
        print()
        print("Next steps:")
        print("1. Build TensorRT engine:")
        print(f"   trtllm-build --checkpoint_dir {args.output_dir} --output_dir ./engines")
        print("2. Run inference:")
        print("   python ../../run.py --engine_dir ./engines --input_text 'Hello, world!'")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        logger.error(f"Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
