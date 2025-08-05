#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import time
import os

# Set environment variable
os.environ['TRTLLM_ENABLE_GEMM_SWIGLU_FUSION'] = '1'

# Import our fusion
from tensorrt_llm._torch.custom_ops.fused_gemm_swiglu import fused_gemm_swiglu_dense

def standard_swiglu(x, weight, bias=None):
    gate_up = torch.nn.functional.linear(x, weight, bias)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

def test_fusion_performance():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # qwen3-1.7b dimensions
    batch_size, seq_len = 16, 1024
    hidden_size = 2048
    intermediate_size = 6144  # gate_up = 2 * intermediate_size = 12288
    
    # Create tensors
    x = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=torch.float16)
    weight = torch.randn(intermediate_size * 2, hidden_size, device=device, dtype=torch.float16)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    
    # Warmup
    for _ in range(10):
        _ = standard_swiglu(x, weight)
        _ = fused_gemm_swiglu_dense(x, weight)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark standard
    start = time.time()
    for _ in range(100):
        result1 = standard_swiglu(x, weight)
    if device == 'cuda':
        torch.cuda.synchronize()
    standard_time = time.time() - start
    
    # Benchmark fused  
    start = time.time()
    for _ in range(100):
        result2 = fused_gemm_swiglu_dense(x, weight)
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = time.time() - start
    
    # Check correctness
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    
    print(f"\nStandard time: {standard_time:.4f}s")
    print(f"Fused time: {fused_time:.4f}s") 
    print(f"Speedup: {standard_time / fused_time:.2f}x")
    print(f"Max diff: {max_diff:.8f}")
    
    return standard_time / fused_time

if __name__ == "__main__":
    speedup = test_fusion_performance()
    print(f"\nFinal speedup: {speedup:.2f}x")