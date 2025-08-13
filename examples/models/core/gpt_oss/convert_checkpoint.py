#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import sys
import time
from pathlib import Path

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.gpt_oss.config import GptOssConfig
from tensorrt_llm.models.gpt_oss.convert import convert_and_save
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Convert GPT-OSS model to TensorRT-LLM checkpoint with native FP8 quantization support")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to the GPT-OSS model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to output TensorRT-LLM checkpoint directory")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Data type for model weights and activations if not quantized (default: float16)")
    parser.add_argument("--tp_size", type=int, default=1,
                       help="Tensor parallelism size (default: 1)")
    parser.add_argument("--pp_size", type=int, default=1,
                       help="Pipeline parallelism size (default: 1)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes (default: 1)")
    parser.add_argument("--load_model_on_cpu", action="store_true",
                       help="Load model on CPU to save GPU memory")
    parser.add_argument("--use_parallel_embedding", action="store_true",
                       help="Use parallel embedding")
    parser.add_argument("--embedding_sharding_dim", type=int, default=0,
                       help="Embedding sharding dimension")
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (auto-detect if not specified)")
    parser.add_argument("--log_level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # FP8 quantization arguments
    parser.add_argument("--use_fp8_qdq", action="store_true", default=False,
                       help="Enable FP8 QDQ quantization (recommended for GPT-OSS MoE)")
    parser.add_argument("--use_mxfp4_fp8", action="store_true", default=False,
                       help="Enable MXFP4 + FP8 quantization (optimized for MoE models)")
    parser.add_argument("--fp8_kv_cache", action="store_true", default=False,
                       help="Enable FP8 KV cache quantization for memory efficiency")

    return parser.parse_args(args)


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    """Create quantization config based on CLI arguments."""
    quant_config = QuantConfig()
    
    # Configure FP8 QDQ quantization (recommended)
    if args.use_fp8_qdq:
        quant_config.quant_algo = QuantAlgo.FP8
        quant_config.clamp_val = [-1200.0, 1200.0]  # Standard FP8 clamp values
        logger.info("Enabled FP8 QDQ quantization (recommended for MoE models)")
    # Configure MXFP4 + FP8 quantization
    elif args.use_mxfp4_fp8:
        quant_config.quant_algo = QuantAlgo.W4A8_MXFP4_FP8
        logger.info("Enabled MXFP4 + FP8 quantization (optimized for MoE models)")
    
    # Configure FP8 KV cache
    if args.fp8_kv_cache:
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        logger.info("Enabled FP8 KV cache quantization")
    
    return quant_config


def convert_checkpoint(args):
    """Convert GPT-OSS checkpoint to TensorRT-LLM format"""
    # Validate paths
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in model directory: {model_dir}")
    
    logger.info(f"Converting GPT-OSS model from {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Tensor parallelism: {args.tp_size}")
    logger.info(f"Pipeline parallelism: {args.pp_size}")
    logger.info(f"Workers: {args.workers}")
    logger.info("Phase 1: Creating BF16 checkpoint (quantization in Phase 2)")
    
    # Create quantization config
    quant_config = args_to_quant_config(args)
    
    start_time = time.time()
    
    # Process each rank
    world_size = args.tp_size * args.pp_size

    for rank in range(world_size):
        logger.info(f"Processing rank {rank}/{world_size}")

        # Create mapping for this rank
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=args.tp_size,
            pp_size=args.pp_size
        )

        # Enforce parallel embedding for TP>1
        if args.tp_size > 1:
            args.use_parallel_embedding = True
            args.embedding_sharding_dim = 0

        # Create config
        config = GptOssConfig.from_hugging_face(
            hf_config_or_dir=str(model_dir),
            dtype=args.dtype,
            mapping=mapping,
            use_parallel_embedding=args.use_parallel_embedding,
            embedding_sharding_dim=args.embedding_sharding_dim,
        )

        # Override vocab_size if specified
        if args.vocab_size is not None:
            config.vocab_size = args.vocab_size

        # Convert and save this rank
        convert_and_save(
            model_dir=str(model_dir),
            output_dir=str(output_dir),
            config=config,
            quant_config=quant_config,
        )

    end_time = time.time()
    logger.info(f"Conversion completed successfully in {end_time - start_time:.2f} seconds!")

    # Print summary
    logger.info("Conversion Summary:")
    logger.info(f"  Model: {model_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Data type: {args.dtype}")
    logger.info(f"  MXFP4+FP8 quantization: {'Enabled' if args.use_mxfp4_fp8 else 'Disabled'}")
    logger.info(f"  FP8 KV cache: {'Enabled' if args.fp8_kv_cache else 'Disabled'}")
    logger.info(f"  Parallelism: TP={args.tp_size}, PP={args.pp_size}")
    logger.info(f"  Total ranks: {world_size}")


def main(args=None):
    """Main entry point"""
    args = parse_arguments(args)
    convert_checkpoint(args)


if __name__ == "__main__":
    main()