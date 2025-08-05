#!/bin/bash

cd /home/ubuntu/build_model

echo "=== Testing SwiGLU Fusion Performance ==="
echo "Model: qwen3-1.7b"
echo

echo "1. Testing WITHOUT SwiGLU Fusion..."
unset TRTLLM_ENABLE_GEMM_SWIGLU_FUSION
python3 run_inference.py --model qwen3-1.7b --backends torch --quantization fp16

echo
echo "2. Testing WITH SwiGLU Fusion..."
export TRTLLM_ENABLE_GEMM_SWIGLU_FUSION=1
python3 run_inference.py --model qwen3-1.7b --backends torch --quantization fp16

echo
echo "=== Test Complete ==="