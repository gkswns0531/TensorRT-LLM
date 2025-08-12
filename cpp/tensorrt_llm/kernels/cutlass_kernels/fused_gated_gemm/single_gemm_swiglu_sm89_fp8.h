#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// FP8 SwiGLU Activation Kernel for SM89 (L4) - 256-bit vectorization optimized
constexpr static int SWIGLU_FP8_ACTIVATION_THREADS_PER_BLOCK = 256;

template <class ActivationOutputType, class GemmOutputType>
__global__ void swiGLUActivationKernelFP8Sm89(
    ActivationOutputType* output, 
    GemmOutputType const* gemm_result,
    ActivationOutputType const* bias_linear, 
    ActivationOutputType const* bias_gate,
    int64_t m, int64_t n, 
    float alpha_scale, float output_scale)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    
    if (token >= m) {
        return;
    }

    output = output + token * n;
    gemm_result = gemm_result + token * n * 2;  // 2x size: [linear, gate]

    // FP8 optimized vectorization: 256-bit = 32 FP8 elements (MoE-inspired coalescing)
    constexpr int64_t FP8_ELEM_PER_THREAD = 32;
    constexpr int64_t WARP_SIZE = 32;
    constexpr int64_t COALESCED_BYTES = WARP_SIZE * FP8_ELEM_PER_THREAD;  // 1024 bytes per warp access
    
    using OutputElem = cutlass::Array<ActivationOutputType, FP8_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, FP8_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FP8_ELEM_PER_THREAD>;
    
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output);
    
    int64_t const start_offset = tid;
    int64_t const stride = SWIGLU_FP8_ACTIVATION_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = n / FP8_ELEM_PER_THREAD;
    int64_t const n_vec = n / FP8_ELEM_PER_THREAD;

    // SiLu activation function for vectorized computation
    cutlass::epilogue::thread::SiLu<ComputeElem> silu_fn{};
    
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        // Load linear and gate values from GEMM result with FP8 → Float conversion
        auto linear_value = cutlass::NumericArrayConverter<float, GemmOutputType, FP8_ELEM_PER_THREAD>{}(
            gemm_result_vec[elem_index]);
        auto gate_value = cutlass::NumericArrayConverter<float, GemmOutputType, FP8_ELEM_PER_THREAD>{}(
            gemm_result_vec[elem_index + n_vec]);
        
        // Apply alpha scaling for FP8 intermediate results (MoE-inspired dynamic scaling)
        // This compensates for FP8 quantization in GEMM computation
        linear_value = linear_value * alpha_scale;
        gate_value = gate_value * alpha_scale;
        
        // Apply bias if provided
        ComputeElem linear_biased = linear_value;
        ComputeElem gate_biased = gate_value;
        
        if (bias_linear) {
            auto bias_l_vec = reinterpret_cast<OutputElem const*>(bias_linear);
            auto bias_l_val = cutlass::NumericArrayConverter<float, ActivationOutputType, FP8_ELEM_PER_THREAD>{}(
                bias_l_vec[elem_index]);
            linear_biased = linear_biased + bias_l_val;
        }
        
        if (bias_gate) {
            auto bias_g_vec = reinterpret_cast<OutputElem const*>(bias_gate);
            auto bias_g_val = cutlass::NumericArrayConverter<float, ActivationOutputType, FP8_ELEM_PER_THREAD>{}(
                bias_g_vec[elem_index]);
            gate_biased = gate_biased + bias_g_val;
        }
        
        // Compute SwiGLU: linear * SiLu(gate)
        auto gate_activated = silu_fn(gate_biased);
        ComputeElem result = linear_biased * gate_activated;
        
        // Apply output scale and convert back to FP8
        result = result * output_scale;
        output_vec[elem_index] = cutlass::NumericArrayConverter<ActivationOutputType, float, FP8_ELEM_PER_THREAD>{}(result);
    }
}

// Single GEMM SwiGLU class for FP8 E4M3 on SM89 (L4)
template <typename ElementA, typename ElementB, typename ElementC, typename ElementD,
          typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD,
          typename ElementAccumulator, typename OperatorClass, typename ArchTag,
          typename ThreadblockShape, typename WarpShape, typename InstructionShape,
          int Stages>
class SingleGemmSwiGLUFP8Sm89
{
public:
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementD;
    using ElementCompute = ElementAccumulator;

    static_assert(std::is_same_v<ElementA, cutlass::float_e4m3_t>, "ElementA must be FP8 E4M3");
    static_assert(std::is_same_v<ElementB, cutlass::float_e4m3_t>, "ElementB must be FP8 E4M3");
    static_assert(std::is_same_v<ElementC, cutlass::float_e4m3_t>, "ElementC must be FP8 E4M3");
    static_assert(std::is_same_v<ElementD, cutlass::float_e4m3_t>, "ElementD must be FP8 E4M3");
    static_assert(std::is_same_v<ElementAccumulator, float>, "Accumulator must be Float32");

    // FP8 E4M3 optimized Single GEMM: A @ B_combined where B_combined = [B_linear, B_gate]
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA,
        cutlass::ComplexTransform::kNone,
        128 / cutlass::sizeof_bits<ElementA>::value,  // FP8 alignment = 128 bits / 8 bits = 16 elements
        ElementB, LayoutB,
        cutlass::ComplexTransform::kNone,
        128 / cutlass::sizeof_bits<ElementB>::value,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,  // SM89 architecture
        ThreadblockShape,
        WarpShape,
        InstructionShape,  // 16x8x32 for FP8 E4M3
        cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        cutlass::arch::OpMultiplyAdd
    >::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversal<GemmKernel>;

    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        cutlass::TensorRef<ElementA const, LayoutA> ref_A;
        cutlass::TensorRef<ElementB const, LayoutB> ref_B;
        cutlass::TensorRef<ElementC const, LayoutC> ref_C;
        cutlass::TensorRef<ElementD, LayoutD> ref_D;
        typename ElementAccumulator alpha;
        typename ElementAccumulator beta;
        float output_scale;     // Compatible with existing interface (reuse for FP8 output scaling)
        
        // Constructor for compatibility with existing dispatch
        Arguments(cutlass::gemm::GemmCoord problem_size_,
                 cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
                 cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
                 cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
                 cutlass::TensorRef<ElementD, LayoutD> ref_D_,
                 typename ElementAccumulator alpha_,
                 typename ElementAccumulator beta_,
                 float output_scale_)
            : problem_size(problem_size_), ref_A(ref_A_), ref_B(ref_B_), ref_C(ref_C_), ref_D(ref_D_),
              alpha(alpha_), beta(beta_), output_scale(output_scale_) {}
    };

    static size_t get_workspace_size(Arguments const& args) {
        // Workspace for GEMM intermediate result (2x output size for linear + gate)
        // FP8 E4M3 = 1 byte per element (50% memory saving vs FP16)
        int m = args.problem_size.m();
        int n_output = args.problem_size.n();
        
        size_t gemm_workspace = m * (n_output * 2) * sizeof(ElementC);  // FP8 intermediate storage
        
        // Add padding for memory alignment (128-bit alignment for FP8)
        size_t alignment = 16;  // 128 bits
        gemm_workspace = (gemm_workspace + alignment - 1) & ~(alignment - 1);
        
        return gemm_workspace;
    }

    cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        
        int m = args.problem_size.m();
        int n_output = args.problem_size.n();
        int k = args.problem_size.k();
        
        // Create problem size for combined GEMM (2x wider output)
        cutlass::gemm::GemmCoord gemm_problem_size(m, n_output * 2, k);
        
        // Workspace for intermediate GEMM result
        ElementC* gemm_output = static_cast<ElementC*>(workspace);
        
        // Setup GEMM arguments with FP8 scaling
        typename Gemm::Arguments gemm_args(
            gemm_problem_size,
            args.ref_A,
            args.ref_B,  // B matrix contains [B_linear, B_gate] concatenated
            args.ref_C,  // Bias can be nullptr
            {gemm_output, n_output * 2},  // Intermediate output
            {args.alpha, args.beta}
        );
        
        // Execute GEMM
        Gemm gemm_op;
        cutlass::Status gemm_status = gemm_op.initialize(gemm_args, workspace);
        if (gemm_status != cutlass::Status::kSuccess) {
            return gemm_status;
        }
        
        gemm_status = gemm_op.run(stream);
        if (gemm_status != cutlass::Status::kSuccess) {
            return gemm_status;
        }
        
        // Launch FP8 SwiGLU activation kernel for SM89
        int64_t blocks = m;
        int64_t threads = SWIGLU_FP8_ACTIVATION_THREADS_PER_BLOCK;
        
        // Extract bias pointers (assume 1D bias vectors)
        ElementD const* bias_linear = nullptr;
        ElementD const* bias_gate = nullptr;
        if (args.ref_C.data()) {
            bias_linear = reinterpret_cast<ElementD const*>(args.ref_C.data());
            bias_gate = bias_linear + n_output;  // Second half
        }
        
        swiGLUActivationKernelFP8Sm89<<<blocks, threads, 0, stream>>>(
            args.ref_D.data(),     // final output
            gemm_output,           // GEMM intermediate result
            bias_linear,           // bias for linear part
            bias_gate,             // bias for gate part  
            m, n_output,           // dimensions
            args.alpha,            // FP8 intermediate scaling (reuse alpha)
            args.output_scale      // final output scaling
        );
        
        return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }
};

} // namespace cutlass_kernels
} // namespace kernels  
} // namespace tensorrt_llm
