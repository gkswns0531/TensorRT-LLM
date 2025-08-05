#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import time
import os

def standard_swiglu_implementation(input_tensor, weight_tensor, bias_tensor=None):
    """기존 구현: 별도 연산들"""
    # GEMM
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Chunk
    gate, up = gate_up.chunk(2, dim=-1)
    
    # SiLU + multiply
    return F.silu(gate) * up

@torch.compile(dynamic=True)
def fused_swiglu_implementation(input_tensor, weight_tensor, bias_tensor=None):
    """우리 구현: torch.compile fused"""
    # GEMM
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Chunk + SiLU + multiply (fused)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

def benchmark_implementations():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    # qwen3-1.7b 실제 크기
    batch_size, seq_len = 16, 1024
    hidden_size = 2048
    intermediate_size = 6144
    gate_up_size = intermediate_size * 2  # 12288
    
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Hidden size: {hidden_size}, Intermediate size: {intermediate_size}")
    print(f"Input shape: ({batch_size * seq_len}, {hidden_size})")
    print(f"Weight shape: ({gate_up_size}, {hidden_size})")
    
    # Create tensors
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size * seq_len, hidden_size, device=device, dtype=torch.float16)
    weight_tensor = torch.randn(gate_up_size, hidden_size, device=device, dtype=torch.float16)
    bias_tensor = torch.randn(gate_up_size, device=device, dtype=torch.float16)
    
    print(f"\nActual tensor shapes:")
    print(f"Input: {input_tensor.shape}")
    print(f"Weight: {weight_tensor.shape}")
    print(f"Bias: {bias_tensor.shape}")
    
    # Warmup (중요: torch.compile 컴파일 시간 제외)
    print(f"\nWarming up...")
    for i in range(20):
        if i % 5 == 0:
            print(f"  Warmup {i+1}/20")
        with torch.no_grad():
            _ = standard_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
            _ = fused_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"\nRunning benchmarks...")
    
    # Test correctness first
    with torch.no_grad():
        result1 = standard_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
        result2 = fused_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
    
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    relative_diff = (max_diff / torch.max(torch.abs(result1)).item()) * 100
    print(f"Correctness check:")
    print(f"  Max absolute diff: {max_diff:.8f}")
    print(f"  Max relative diff: {relative_diff:.6f}%")
    
    if max_diff > 1e-3:
        print(f"  ⚠️  WARNING: Large difference detected!")
    else:
        print(f"  ✅ Results match within tolerance")
    
    # Benchmark standard implementation
    print(f"\nBenchmarking standard implementation...")
    n_iterations = 500
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            result1 = standard_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    standard_time = time.time() - start_time
    
    # Benchmark fused implementation
    print(f"Benchmarking fused implementation...")
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            result2 = fused_swiglu_implementation(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    fused_time = time.time() - start_time
    
    # Results
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({n_iterations} iterations)")
    print(f"{'='*60}")
    print(f"Standard implementation: {standard_time:.4f}s ({standard_time/n_iterations*1000:.3f}ms per call)")
    print(f"Fused implementation:    {fused_time:.4f}s ({fused_time/n_iterations*1000:.3f}ms per call)")
    print(f"")
    
    if fused_time < standard_time:
        speedup = standard_time / fused_time
        improvement = ((standard_time - fused_time) / standard_time) * 100
        print(f"🚀 Speedup: {speedup:.2f}x ({improvement:.1f}% faster)")
    else:
        slowdown = fused_time / standard_time  
        regression = ((fused_time - standard_time) / standard_time) * 100
        print(f"🐌 Slowdown: {slowdown:.2f}x ({regression:.1f}% slower)")
    
    print(f"{'='*60}")
    
    return standard_time, fused_time

if __name__ == "__main__":
    try:
        standard_time, fused_time = benchmark_implementations()
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()