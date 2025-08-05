#!/usr/bin/env python3
"""
TensorRT-LLM Torch Backend SwiGLU Fusion Performance Test

This script tests the performance improvement of our custom SwiGLU fusion implementation
by comparing:
1. Standard path (no fusion): gate_up_proj -> chunk -> silu -> mul
2. Fusion path (with fusion): fused_gemm_swiglu_dense (GEMM+SwiGLU in one kernel)
"""

import time
import torch
import os
import gc
import warnings
from typing import List, Dict, Any


def clean_vram():
    """Clean VRAM and wait for complete memory release"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("VRAM cleanup completed, waiting 3 seconds...")
    time.sleep(3)


def setup_trtllm_torch(model_name: str, enable_fusion: bool = False) -> Any:
    """Setup TensorRT-LLM Torch backend with optional SwiGLU fusion"""
    print(f"Setting up TensorRT-LLM Torch backend (fusion={enable_fusion})...")
    
    # Import TensorRT-LLM
    from tensorrt_llm import LLM
    
    try:
        llm = LLM(
            model=f"/workspace/build_model/{model_name}",
            backend="pytorch",
            max_num_tokens=8192,  # Smaller for testing
            max_batch_size=4,     # Smaller batch for stability
        )
        
        # Enable fusion by modifying model layers
        if enable_fusion:
            print("🔧 Enabling SwiGLU fusion...")
            enable_fusion_count = 0
            
            try:
                # Access the torch model
                torch_model = llm.runtime_context.model
                
                # Find and enable fusion in decoder layers
                if hasattr(torch_model, 'layers'):
                    for i, layer in enumerate(torch_model.layers):
                        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'enable_fused_gemm_swiglu'):
                            layer.mlp.enable_fused_gemm_swiglu = True
                            enable_fusion_count += 1
                            
                elif hasattr(torch_model, 'transformer') and hasattr(torch_model.transformer, 'layers'):
                    for i, layer in enumerate(torch_model.transformer.layers):
                        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'enable_fused_gemm_swiglu'):
                            layer.mlp.enable_fused_gemm_swiglu = True
                            enable_fusion_count += 1
                            
                print(f"✅ Enabled fusion in {enable_fusion_count} layers")
                
            except Exception as e:
                print(f"⚠️  Failed to enable fusion: {e}")
                print("   Fusion will be disabled for this test")
        
        return llm
        
    except Exception as e:
        print(f"❌ Failed to setup TensorRT-LLM: {e}")
        return None


def measure_inference_performance(model: Any, prompts: List[str], backend_name: str) -> Dict[str, float]:
    """Measure inference performance with detailed timing"""
    print(f"\n🚀 Testing {backend_name}...")
    
    from tensorrt_llm import SamplingParams
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.8,
        top_p=0.9,
        min_tokens=100  # Ensure consistent generation length
    )
    
    # Warmup
    print("  🔥 Warming up...")
    try:
        warmup_outputs = model.generate(prompts[:1], sampling_params)
        clean_vram()
    except Exception as e:
        print(f"  ⚠️  Warmup failed: {e}")
    
    # Measure Time To First Token (TTFT)
    print("  ⏱️  Measuring TTFT...")
    torch.cuda.synchronize()
    ttft_start = time.perf_counter()
    
    try:
        ttft_sampling = SamplingParams(max_tokens=1, temperature=0.0)
        ttft_outputs = model.generate(prompts, ttft_sampling)
        torch.cuda.synchronize()
        ttft_end = time.perf_counter()
        ttft_ms = (ttft_end - ttft_start) * 1000
        
    except Exception as e:
        print(f"  ❌ TTFT measurement failed: {e}")
        ttft_ms = float('inf')
    
    # Measure full generation
    print("  📊 Measuring full generation...")
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    
    try:
        outputs = model.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        total_tps = total_tokens / total_time
        
        result = {
            'backend': backend_name,
            'ttft_ms': ttft_ms,
            'total_time': total_time,
            'total_tokens': total_tokens,
            'tps': total_tps,
            'prompts_count': len(prompts),
            'avg_tokens_per_prompt': total_tokens / len(prompts)
        }
        
        print(f"  ✅ Results: TTFT={ttft_ms:.2f}ms, TPS={total_tps:.2f}, Tokens={total_tokens}")
        return result
        
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return {
            'backend': backend_name,
            'ttft_ms': float('inf'),
            'total_time': float('inf'),
            'total_tokens': 0,
            'tps': 0.0,
            'prompts_count': len(prompts),
            'avg_tokens_per_prompt': 0
        }


def test_fusion_functionality():
    """Test basic fusion functionality"""
    print("\n🧪 Testing SwiGLU Fusion Functionality...")
    
    try:
        from tensorrt_llm._torch.custom_ops import (
            fused_gemm_swiglu_dense, 
            can_use_gemm_swiglu_fusion,
            get_fusion_info
        )
        
        # Test on GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        
        # Create test tensors
        batch_size, seq_len, hidden_size = 2, 128, 1024
        intermediate_size = 2048
        
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
        weight_tensor = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
        bias_tensor = torch.randn(intermediate_size, dtype=dtype, device=device)
        
        # Check fusion capability
        can_fuse = can_use_gemm_swiglu_fusion(input_tensor, weight_tensor, bias_tensor)
        fusion_info = get_fusion_info(input_tensor, weight_tensor)
        
        print(f"  📋 Fusion Info: {fusion_info}")
        print(f"  ✅ Can fuse: {can_fuse}")
        
        if can_fuse:
            # Test fusion
            with torch.no_grad():
                fused_result = fused_gemm_swiglu_dense(input_tensor, weight_tensor, bias_tensor)
                
                # Compare with standard implementation
                gate_up = torch.nn.functional.linear(input_tensor, weight_tensor.t(), bias_tensor)
                gate, up = gate_up.chunk(2, dim=-1)
                standard_result = torch.nn.functional.silu(gate) * up
                
                # Check numerical accuracy
                max_diff = torch.max(torch.abs(fused_result - standard_result)).item()
                rel_diff = (torch.norm(fused_result - standard_result) / torch.norm(standard_result)).item()
                
                print(f"  📏 Max difference: {max_diff:.6f}")
                print(f"  📐 Relative difference: {rel_diff:.6f}")
                
                if max_diff < 1e-3 and rel_diff < 1e-3:
                    print(f"  ✅ Fusion accuracy test PASSED!")
                    return True
                else:
                    print(f"  ❌ Fusion accuracy test FAILED!")
                    return False
        else:
            print(f"  ⚠️  Fusion not supported on this configuration")
            return False
            
    except Exception as e:
        print(f"  ❌ Fusion functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 TensorRT-LLM Torch Backend SwiGLU Fusion Performance Test")
    print("=" * 80)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_properties(0).name
    print(f"🔧 GPU: {gpu_name}")
    
    # Test basic fusion functionality first
    if not test_fusion_functionality():
        print("\n❌ Basic fusion test failed, skipping performance test")
        return
    
    # Test prompts
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis.",
        "How does the internet work?"
    ]
    
    model_name = "qwen3-1.7b"
    results = []
    
    try:
        # Test 1: Standard path (no fusion)
        print(f"\n🔧 Test 1: Standard Path (No Fusion)")
        model_standard = setup_trtllm_torch(model_name, enable_fusion=False)
        if model_standard:
            result_standard = measure_inference_performance(
                model_standard, test_prompts, "TensorRT-LLM Torch (Standard)"
            )
            results.append(result_standard)
            del model_standard
        clean_vram()
        
        # Test 2: Fusion path
        print(f"\n🔧 Test 2: Fusion Path (SwiGLU Fusion)")
        model_fused = setup_trtllm_torch(model_name, enable_fusion=True)
        if model_fused:
            result_fused = measure_inference_performance(
                model_fused, test_prompts, "TensorRT-LLM Torch (Fusion)"
            )
            results.append(result_fused)
            del model_fused
        clean_vram()
        
        # Results comparison
        if len(results) >= 2:
            print(f"\n{'='*80}")
            print("🎯 SWIGLU FUSION PERFORMANCE COMPARISON")
            print(f"{'='*80}")
            print(f"{'Backend':<30} {'TTFT (ms)':<12} {'TPS':<12} {'Tokens':<8} {'Time (s)':<10}")
            print(f"{'-'*80}")
            
            for result in results:
                print(f"{result['backend']:<30} "
                      f"{result['ttft_ms']:<12.2f} "
                      f"{result['tps']:<12.2f} "
                      f"{result['total_tokens']:<8} "
                      f"{result['total_time']:<10.2f}")
            
            # Calculate improvements
            if len(results) == 2:
                standard, fused = results
                if standard['tps'] > 0 and fused['tps'] > 0:
                    tps_improvement = (fused['tps'] / standard['tps'] - 1) * 100
                    ttft_improvement = (standard['ttft_ms'] / fused['ttft_ms'] - 1) * 100 if fused['ttft_ms'] > 0 else 0
                    
                    print(f"\n🚀 Performance Improvements:")
                    print(f"   Throughput (TPS): {tps_improvement:+.1f}%")
                    print(f"   TTFT: {ttft_improvement:+.1f}%")
                    
                    if tps_improvement > 5:
                        print(f"   ✅ Significant fusion speedup achieved!")
                    elif tps_improvement > 0:
                        print(f"   ✅ Modest fusion improvement achieved!")
                    else:
                        print(f"   ⚠️  No significant speedup (may need larger workload)")
        
        print(f"\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()