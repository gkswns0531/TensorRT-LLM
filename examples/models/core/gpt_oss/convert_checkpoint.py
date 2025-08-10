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


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Convert GPT-OSS model to TensorRT-LLM checkpoint")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to the GPT-OSS model directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to output TensorRT-LLM checkpoint directory")
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type for conversion (default: float16)")
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

    # MXFP4/BF16 MoE export control
    parser.add_argument("--moe_export", type=str, default="auto",
                        choices=["auto", "mxfp4", "bf16"],
                        help="MoE export mode: 'mxfp4' keeps FP4 blocks and NVFP4 scales; 'bf16' dequantizes to BF16; 'auto' based on target_arch")
    parser.add_argument("--target_arch", type=str, default=None,
                        help="Target architecture hint (e.g., sm80, a100, sm89, l4, sm100, h100, b200). Used when moe_export=auto")

    # Advanced options
    parser.add_argument("--nvfp4_scale_mode", type=str, default="heuristic",
                        choices=["heuristic", "ones", "auto"],
                        help="NVFP4 aux scale generation mode for MXFP4 export")
    parser.add_argument("--stream_tile_rows", type=int, default=1024,
                        help="Tile rows for streaming dequant when exporting BF16")
    parser.add_argument("--interleave_scales", action="store_true",
                        help="If set, interleave NVFP4 scales at convert-time (else loader interleaves)")
    return parser.parse_args(args)


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
            quant_config=None,
            moe_export=args.moe_export,
            target_arch=args.target_arch,
            nvfp4_scale_mode=args.nvfp4_scale_mode,
            stream_tile_rows=args.stream_tile_rows,
            interleave_scales=args.interleave_scales,
        )

    end_time = time.time()
    logger.info(f"Conversion completed successfully in {end_time - start_time:.2f} seconds!")

    # Print summary
    logger.info("Conversion Summary:")
    logger.info(f"  Model: {model_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Data type: {args.dtype}")
    logger.info(f"  Parallelism: TP={args.tp_size}, PP={args.pp_size}")
    logger.info(f"  Total ranks: {world_size}")


def main(args=None):
    """Main entry point"""
    args = parse_arguments(args)
    convert_checkpoint(args)


if __name__ == "__main__":
    main()