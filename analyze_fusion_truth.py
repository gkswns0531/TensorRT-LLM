#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import time

def standard_swiglu(input_tensor, weight_tensor, bias_tensor=None):
    """기존 구현: 별도 연산들"""
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

@torch.compile(dynamic=True)
def compiled_swiglu(input_tensor, weight_tensor, bias_tensor=None):
    """torch.compile 버전"""
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up

def true_fused_swiglu_attempt(input_tensor, weight_tensor, bias_tensor=None):
    """진짜 fusion 시도: 중간 결과 없이"""
    # 이론적으로는 이렇게 해야 진짜 fusion
    # weight를 반으로 나누어서 gate_weight, up_weight 분리
    hidden_size = weight_tensor.shape[1]
    gate_up_size = weight_tensor.shape[0]
    intermediate_size = gate_up_size // 2
    
    gate_weight = weight_tensor[:intermediate_size, :]  # [6144, 2048]
    up_weight = weight_tensor[intermediate_size:, :]    # [6144, 2048]
    
    gate_bias = bias_tensor[:intermediate_size] if bias_tensor is not None else None
    up_bias = bias_tensor[intermediate_size:] if bias_tensor is not None else None
    
    # 별도 GEMM으로 gate와 up 계산
    gate = torch.nn.functional.linear(input_tensor, gate_weight, gate_bias)
    up = torch.nn.functional.linear(input_tensor, up_weight, up_bias)
    
    # SiLU + multiply (중간 gate_up 텐서 생성 없음)
    return F.silu(gate) * up

def analyze_compilation():
    """torch.compile이 실제로 뭘 하는지 분석"""
    device = 'cuda'
    
    # 작은 테스트 케이스
    batch_size = 4
    hidden_size = 128
    intermediate_size = 256
    gate_up_size = intermediate_size * 2
    
    input_tensor = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
    weight_tensor = torch.randn(gate_up_size, hidden_size, device=device, dtype=torch.float16)
    bias_tensor = torch.randn(gate_up_size, device=device, dtype=torch.float16)
    
    print("=== Compilation Analysis ===")
    
    # 1. Standard 실행
    print("1. Standard implementation:")
    with torch.no_grad():
        result1 = standard_swiglu(input_tensor, weight_tensor, bias_tensor)
    print(f"   Output shape: {result1.shape}")
    
    # 2. Compiled 실행 (첫 번째는 컴파일 시간 포함)
    print("2. Compiled implementation (first run - includes compilation):")
    start = time.time()
    with torch.no_grad():
        result2 = compiled_swiglu(input_tensor, weight_tensor, bias_tensor)
    compile_time = time.time() - start
    print(f"   Output shape: {result2.shape}")
    print(f"   First run time (with compilation): {compile_time:.4f}s")
    
    # 3. True fused 실행
    print("3. True fused implementation:")
    with torch.no_grad():
        result3 = true_fused_swiglu_attempt(input_tensor, weight_tensor, bias_tensor)
    print(f"   Output shape: {result3.shape}")
    
    # 정확성 체크
    print("\n=== Correctness Check ===")
    diff12 = torch.max(torch.abs(result1 - result2)).item()
    diff13 = torch.max(torch.abs(result1 - result3)).item()
    print(f"Standard vs Compiled max diff: {diff12:.8f}")
    print(f"Standard vs True-fused max diff: {diff13:.8f}")
    
    # 메모리 사용량 분석
    print("\n=== Memory Analysis ===")
    
    def measure_memory(func, name):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = func(input_tensor, weight_tensor, bias_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"{name} peak memory: {peak_memory:.2f} MB")
        return peak_memory
    
    mem1 = measure_memory(standard_swiglu, "Standard")
    mem2 = measure_memory(compiled_swiglu, "Compiled") 
    mem3 = measure_memory(true_fused_swiglu_attempt, "True-fused")
    
    print(f"\nMemory comparison:")
    print(f"  Compiled vs Standard: {((mem2-mem1)/mem1)*100:.1f}% difference")
    print(f"  True-fused vs Standard: {((mem3-mem1)/mem1)*100:.1f}% difference")
    
    # 큰 텐서로 성능 테스트
    print("\n=== Performance Test (Larger Scale) ===")
    
    batch_size = 16 * 1024
    hidden_size = 2048
    intermediate_size = 6144
    gate_up_size = intermediate_size * 2
    
    input_tensor = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
    weight_tensor = torch.randn(gate_up_size, hidden_size, device=device, dtype=torch.float16)
    bias_tensor = torch.randn(gate_up_size, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = standard_swiglu(input_tensor, weight_tensor, bias_tensor)
            _ = compiled_swiglu(input_tensor, weight_tensor, bias_tensor)
            _ = true_fused_swiglu_attempt(input_tensor, weight_tensor, bias_tensor)
    
    torch.cuda.synchronize()
    
    def benchmark_func(func, name, iterations=100):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = func(input_tensor, weight_tensor, bias_tensor)
        torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / iterations * 1000  # ms
        print(f"{name}: {avg_time:.3f}ms per call")
        return avg_time
    
    time1 = benchmark_func(standard_swiglu, "Standard")
    time2 = benchmark_func(compiled_swiglu, "Compiled")
    time3 = benchmark_func(true_fused_swiglu_attempt, "True-fused")
    
    print(f"\nSpeedups:")
    print(f"  Compiled vs Standard: {time1/time2:.2f}x")
    print(f"  True-fused vs Standard: {time1/time3:.2f}x")
    print(f"  True-fused vs Compiled: {time2/time3:.2f}x")

if __name__ == "__main__":
    analyze_compilation()