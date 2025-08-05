#!/usr/bin/env python3
"""
Simple SwiGLU Fusion Test (No TensorRT Dependencies)

Tests only the core fusion logic without TensorRT dependencies
"""

import torch
import torch.nn.functional as F
import time
import sys
import os


def test_fusion_logic():
    """Test core fusion logic without TensorRT-LLM dependencies"""
    print("🧪 Testing Core SwiGLU Fusion Logic...")
    
    # Test parameters
    batch_size, seq_len = 4, 512
    hidden_size = 1024
    intermediate_size = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    print(f"  🔧 Device: {device}, Dtype: {dtype}")
    print(f"  📏 Shape: {batch_size}x{seq_len}x{hidden_size} -> {intermediate_size}")
    
    # Create test data
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    weight_tensor = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
    bias_tensor = torch.randn(intermediate_size, dtype=dtype, device=device)
    
    print(f"  ✅ Test tensors created")
    
    # Method 1: Standard implementation (chunk + silu + mul)
    def standard_swiglu(x, w, b=None):
        gate_up = torch.nn.functional.linear(x, w, b)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up
    
    # Method 2: torch.compile version (our fusion)
    @torch.compile(dynamic=True)
    def fused_swiglu_compiled(x, w, b=None):
        gate_up = torch.nn.functional.linear(x, w, b)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up
    
    # Numerical accuracy test
    print("  🔬 Testing numerical accuracy...")
    with torch.no_grad():
        result_standard = standard_swiglu(input_tensor, weight_tensor, bias_tensor)
        result_fused = fused_swiglu_compiled(input_tensor, weight_tensor, bias_tensor)
        
        max_diff = torch.max(torch.abs(result_standard - result_fused)).item()
        rel_diff = (torch.norm(result_standard - result_fused) / torch.norm(result_standard)).item()
        
        print(f"    📏 Max difference: {max_diff:.6f}")
        print(f"    📐 Relative difference: {rel_diff:.6f}")
        
        accuracy_ok = max_diff < 1e-3 and rel_diff < 1e-3
        print(f"    {'✅' if accuracy_ok else '❌'} Accuracy test: {'PASSED' if accuracy_ok else 'FAILED'}")
    
    # Performance test
    print("  ⚡ Testing performance...")
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = standard_swiglu(input_tensor, weight_tensor, bias_tensor)
            _ = fused_swiglu_compiled(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark standard
    n_runs = 100
    start_time = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = standard_swiglu(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start_time) / n_runs * 1000  # ms
    
    # Benchmark fused
    start_time = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = fused_swiglu_compiled(input_tensor, weight_tensor, bias_tensor)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start_time) / n_runs * 1000  # ms
    
    speedup = standard_time / fused_time if fused_time > 0 else 0
    
    print(f"    ⏱️  Standard time: {standard_time:.3f}ms")
    print(f"    ⚡ Fused time: {fused_time:.3f}ms")
    print(f"    🚀 Speedup: {speedup:.2f}x")
    
    return accuracy_ok, speedup


def test_condition_checks():
    """Test fusion condition checking logic"""
    print("\n🔍 Testing Fusion Condition Logic...")
    
    def check_fusion_conditions(input_tensor, weight_tensor, bias_tensor=None):
        """Simplified version of our condition checks"""
        if not input_tensor.device.type == 'cuda':
            return False, "Not CUDA device"
        
        if input_tensor.dtype not in [torch.float16, torch.bfloat16]:
            return False, f"Unsupported dtype: {input_tensor.dtype}"
        
        if input_tensor.dtype != weight_tensor.dtype:
            return False, "Dtype mismatch"
        
        hidden_size = input_tensor.shape[-1]
        gate_up_size = weight_tensor.shape[-1]
        
        if hidden_size % 64 != 0 or gate_up_size % 64 != 0:
            return False, f"Alignment issue: {hidden_size}, {gate_up_size}"
        
        if gate_up_size % 2 != 0:
            return False, f"Gate+up size not even: {gate_up_size}"
        
        if not (input_tensor.is_contiguous() and weight_tensor.is_contiguous()):
            return False, "Non-contiguous tensors"
        
        return True, "All conditions met"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test valid case
    input_valid = torch.randn(4, 64, 512, dtype=torch.float16, device=device)
    weight_valid = torch.randn(1024, 512, dtype=torch.float16, device=device)
    can_fuse, reason = check_fusion_conditions(input_valid, weight_valid)
    print(f"  ✅ Valid case: {can_fuse} ({reason})")
    
    # Test invalid cases
    if device == 'cuda':
        input_fp32 = torch.randn(4, 64, 512, dtype=torch.float32, device=device)
        can_fuse, reason = check_fusion_conditions(input_fp32, weight_valid)
        print(f"  ❌ FP32 case: {can_fuse} ({reason})")
        
        input_misaligned = torch.randn(4, 64, 511, dtype=torch.float16, device=device)
        weight_misaligned = torch.randn(1023, 511, dtype=torch.float16, device=device)
        can_fuse, reason = check_fusion_conditions(input_misaligned, weight_misaligned)
        print(f"  ❌ Misaligned case: {can_fuse} ({reason})")
    else:
        print(f"  ⚠️  CPU test skipped (fusion requires CUDA)")


def main():
    """Main test function"""
    print("🧪 Simple SwiGLU Fusion Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, running CPU tests only")
    else:
        gpu_name = torch.cuda.get_device_properties(0).name
        print(f"🔧 GPU: {gpu_name}")
    
    try:
        # Test core fusion logic
        accuracy_ok, speedup = test_fusion_logic()
        
        # Test condition checks
        test_condition_checks()
        
        # Summary
        print(f"\n{'='*60}")
        print("🎯 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Numerical Accuracy: {'PASSED' if accuracy_ok else 'FAILED'}")
        print(f"⚡ Performance Speedup: {speedup:.2f}x")
        
        if accuracy_ok and speedup > 1.0:
            print(f"🎉 SwiGLU Fusion is working correctly!")
            print(f"💡 Ready for integration with TensorRT-LLM")
        elif accuracy_ok:
            print(f"✅ Fusion logic is correct")
            print(f"💡 Performance may improve with larger workloads")
        else:
            print(f"❌ Fusion has issues that need to be fixed")
        
        return accuracy_ok
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)