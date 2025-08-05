"""
GEMM+SwiGLU Fusion for Torch Backend

Following TensorRT-LLM conventions:
- torch.compile for fusion (like MOE)
- FlashInfer integration
- Custom op registration for graph break prevention
"""

import torch
import torch.nn.functional as F
from tensorrt_llm._utils import get_sm_version


def _can_fuse_gemm_swiglu(input_tensor: torch.Tensor, weight_tensor: torch.Tensor, 
                         bias_tensor: torch.Tensor = None) -> bool:
    """
    Check if GEMM+SwiGLU fusion is possible
    Based on validated MOE GEMM conditions
    """
    if not input_tensor.device.type == 'cuda':
        return False
    
    # Check SM version (SM80+ required)
    try:
        sm_version = get_sm_version()
        if sm_version < 80:
            return False
    except:
        return False
    
    # Check dtype (FP16/BF16 only, no FP8/FP32)
    if input_tensor.dtype not in [torch.float16, torch.bfloat16]:
        return False
    
    if input_tensor.dtype != weight_tensor.dtype:
        return False
    
    # Check alignment (MOE GEMM requirements)
    hidden_size = input_tensor.shape[-1]
    gate_up_size = weight_tensor.shape[-1]
    
    if hidden_size % 64 != 0 or gate_up_size % 64 != 0:
        return False
    
    # Check gate_up_size is even (for gate+up split)
    if gate_up_size % 2 != 0:
        return False
    
    # Check contiguous tensors
    if not (input_tensor.is_contiguous() and weight_tensor.is_contiguous()):
        return False
    
    # Check bias compatibility if present
    if bias_tensor is not None:
        if (bias_tensor.dtype != input_tensor.dtype or 
            bias_tensor.shape[0] != gate_up_size or
            not bias_tensor.is_contiguous()):
            return False
    
    return True


@torch.compile(dynamic=True)
def _fused_gemm_swiglu_compiled(input_tensor: torch.Tensor, 
                               weight_tensor: torch.Tensor,
                               bias_tensor: torch.Tensor = None) -> torch.Tensor:
    """
    GEMM+SwiGLU fusion using torch.compile (following MOE pattern)
    
    This follows the exact same pattern as MOE's swiglu_fused_moe but for dense layers
    """
    # GEMM operation
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Split into gate and up (Dense model order: gate first, up second)
    gate, up = gate_up.chunk(2, dim=-1)
    
    # SwiGLU: SiLU(gate) * up
    return F.silu(gate) * up


def _fallback_gemm_swiglu(input_tensor: torch.Tensor, 
                         weight_tensor: torch.Tensor,
                         bias_tensor: torch.Tensor = None) -> torch.Tensor:
    """
    Fallback implementation without fusion
    """
    # GEMM operation
    gate_up = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Apply SwiGLU activation (same as existing gated_mlp.swiglu)
    from ..custom_ops import IS_FLASHINFER_AVAILABLE
    
    if IS_FLASHINFER_AVAILABLE:
        from ..custom_ops import flashinfer_silu_and_mul
        return flashinfer_silu_and_mul(gate_up)
    else:
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up


# Custom op registration for graph break prevention (following flashinfer pattern)
@torch.library.custom_op("trtllm::fused_gemm_swiglu_dense", mutates_args=())
def fused_gemm_swiglu_dense(input: torch.Tensor,
                           weight: torch.Tensor,
                           bias: torch.Tensor = None) -> torch.Tensor:
    """
    Custom op wrapper for GEMM+SwiGLU fusion to prevent graph breaks
    
    Following TensorRT-LLM convention like flashinfer_silu_and_mul
    """
    if _can_fuse_gemm_swiglu(input, weight, bias):
        return _fused_gemm_swiglu_compiled(input, weight, bias)
    else:
        return _fallback_gemm_swiglu(input, weight, bias)


@fused_gemm_swiglu_dense.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Fake implementation for shape inference and torch.compile"""
    batch_dims = input.shape[:-1]
    intermediate_size = weight.shape[-1] // 2  # SwiGLU output is half size
    output_shape = batch_dims + (intermediate_size,)
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


def can_use_gemm_swiglu_fusion(input_tensor: torch.Tensor, 
                              weight_tensor: torch.Tensor,
                              bias_tensor: torch.Tensor = None) -> bool:
    """
    Public interface to check fusion capability
    """
    return _can_fuse_gemm_swiglu(input_tensor, weight_tensor, bias_tensor)


def get_fusion_info(input_tensor: torch.Tensor, weight_tensor: torch.Tensor) -> dict:
    """
    Get fusion configuration info for debugging/monitoring
    """
    batch_dims = input_tensor.shape[:-1]
    hidden_size = input_tensor.shape[-1]
    gate_up_size = weight_tensor.shape[-1]
    intermediate_size = gate_up_size // 2
    
    m = torch.numel(torch.tensor(batch_dims)) if batch_dims else 1
    
    return {
        'can_fuse': _can_fuse_gemm_swiglu(input_tensor, weight_tensor),
        'batch_size': m,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'sm_version': get_sm_version(),
        'dtype': str(input_tensor.dtype),
        'memory_gb': (input_tensor.numel() + weight_tensor.numel()) * input_tensor.element_size() / 1e9
    }