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

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// SwiGLU Activation Kernel for SM89 (L4) - adapted from A100 implementation
constexpr static int SWIGLU_ACTIVATION_THREADS_PER_BLOCK_SM89 = 256;

template <class ActivationOutputType, class GemmOutputType>
__global__ void swiGLUActivationKernelSm89(ActivationOutputType* output, GemmOutputType const* gemm_result,
    ActivationOutputType const* bias_linear, ActivationOutputType const* bias_gate,
    int64_t m, int64_t n, float output_scale)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    
    if (token >= m) {
        return;
    }

    output = output + token * n;
    gemm_result = gemm_result + token * n * 2;  // 2x size: [linear, gate]

    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / (sizeof(ActivationOutputType) * 8);

    using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output);
    
    int64_t const start_offset = tid;
    int64_t const stride = SWIGLU_ACTIVATION_THREADS_PER_BLOCK_SM89;
    int64_t const num_elems_in_col = n / ACTIVATION_ELEM_PER_THREAD;
    int64_t const n_vec = n / ACTIVATION_ELEM_PER_THREAD;

    // SiLu activation function
    cutlass::epilogue::thread::SiLu<ComputeElem> silu_fn{};
    
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        // Load linear and gate values from GEMM result
        auto linear_value = cutlass::NumericArrayConverter<float, GemmOutputType, ACTIVATION_ELEM_PER_THREAD>{}(
            gemm_result_vec[elem_index]);
        auto gate_value = cutlass::NumericArrayConverter<float, GemmOutputType, ACTIVATION_ELEM_PER_THREAD>{}(
            gemm_result_vec[elem_index + n_vec]);
        
        // Apply bias if provided
        ComputeElem linear_biased = linear_value;
        ComputeElem gate_biased = gate_value;
        
        if (bias_linear) {
            auto bias_l_vec = reinterpret_cast<OutputElem const*>(bias_linear);
            auto bias_l_val = cutlass::NumericArrayConverter<float, ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>{}(
                bias_l_vec[elem_index]);
            linear_biased = linear_biased + bias_l_val;
        }
        
        if (bias_gate) {
            auto bias_g_vec = reinterpret_cast<OutputElem const*>(bias_gate);
            auto bias_g_val = cutlass::NumericArrayConverter<float, ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>{}(
                bias_g_vec[elem_index]);
            gate_biased = gate_biased + bias_g_val;
        }
        
        // Compute SwiGLU: linear * SiLu(gate)
        auto gate_activated = silu_fn(gate_biased);
        ComputeElem result = linear_biased * gate_activated;
        
        // Apply output scale
        result = result * output_scale;
        
        // Convert and store result
        output_vec[elem_index] = cutlass::NumericArrayConverter<ActivationOutputType, float, ACTIVATION_ELEM_PER_THREAD>{}(result);
    }
}

// Single GEMM SwiGLU class for SM89 (L4) - optimized from MoE approach
template <typename ElementA, typename ElementB, typename ElementC, typename ElementD,
          typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD,
          typename ElementAccumulator, typename OperatorClass, typename ArchTag,
          typename ThreadblockShape, typename WarpShape, typename InstructionShape,
          int Stages>
class SingleGemmSwiGLUSm89
{
public:
    using ElementInputA = ElementA;
    using ElementInputB = ElementB;
    using ElementOutput = ElementD;
    using ElementCompute = ElementAccumulator;

    // Single GEMM that computes A @ B_combined where B_combined = [B_linear, B_gate]
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA,
        cutlass::ComplexTransform::kNone,
        128 / cutlass::sizeof_bits<ElementA>::value,
        ElementB, LayoutB,
        cutlass::ComplexTransform::kNone,
        128 / cutlass::sizeof_bits<ElementB>::value,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,  // SM89 architecture
        ThreadblockShape,
        WarpShape,
        InstructionShape,
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
        float output_scale;
    };

    static size_t get_workspace_size(Arguments const& args) {
        // Workspace for GEMM intermediate result (2x output size for linear + gate)
        int m = args.problem_size.m();
        int n_output = args.problem_size.n();
        return m * (n_output * 2) * sizeof(ElementC);
    }

    cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        
        int m = args.problem_size.m();
        int n_output = args.problem_size.n();
        int k = args.problem_size.k();
        
        // Create problem size for combined GEMM (2x wider output)
        cutlass::gemm::GemmCoord gemm_problem_size(m, n_output * 2, k);
        
        // Workspace for intermediate GEMM result
        ElementC* gemm_output = static_cast<ElementC*>(workspace);
        
        // Setup GEMM arguments
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
        
        // Launch SwiGLU activation kernel for SM89
        int64_t blocks = m;
        int64_t threads = SWIGLU_ACTIVATION_THREADS_PER_BLOCK_SM89;
        
        // Extract bias pointers (assume 1D bias vectors)
        ElementD const* bias_linear = nullptr;
        ElementD const* bias_gate = nullptr;
        if (args.ref_C.data()) {
            bias_linear = reinterpret_cast<ElementD const*>(args.ref_C.data());
            bias_gate = bias_linear + n_output;  // Second half
        }
        
        swiGLUActivationKernelSm89<<<blocks, threads, 0, stream>>>(
            args.ref_D.data(),  // final output
            gemm_output,        // GEMM intermediate result
            bias_linear,        // bias for linear part
            bias_gate,          // bias for gate part  
            m, n_output,        // dimensions
            args.output_scale   // output scaling
        );
        
        return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }
};

} // namespace cutlass_kernels
} // namespace kernels  
} // namespace tensorrt_llm
