#pragma once

#include "single_gemm_swiglu_sm89_fp8.h"
#include "cutlass/gemm/gemm.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// DeviceGemmGatedSm89 wrapper for FP8 Single GEMM SwiGLU - optimized for L4 (SM89) FP8
template <typename ElementType, typename AccumElementType, typename CTAShape, typename WarpShape_, typename ClusterShape,
          bool SwapAB = false>
struct DeviceGemmGatedSm89SingleFP8
{
    static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8 E4M3 for this specialization");
    
    using ElementA = ElementType;
    using ElementB = ElementType;
    using ElementC = ElementType;
    using ElementD = ElementType;
    using ElementAccumulator = AccumElementType;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using ThreadblockShape = CTAShape;
    using WarpShape = WarpShape_;
    // FP8 E4M3 uses 16x8x32 instruction shape for optimal performance
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

    // FP8 Single GEMM SwiGLU operation for SM89 (L4)
    using SingleGemm = SingleGemmSwiGLUFP8Sm89<
        ElementA, ElementB, ElementC, ElementD,
        LayoutA, LayoutB, LayoutC, LayoutD,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,  // SM89 architecture tag for L4
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        4  // Stages - optimal for SM89 FP8
    >;

    using Arguments = typename SingleGemm::Arguments;

    static bool can_implement(Arguments const& args) {
        // Check alignment requirements for SM89 FP8
        int m = args.problem_size.m();
        int n = args.problem_size.n();
        int k = args.problem_size.k();

        // FP8 alignment checks - stricter requirements
        // K dimension must be multiple of 32 for 16x8x32 instruction
        if (k % 32 != 0 || n % 32 != 0) {
            return false;
        }

        // Check tensor alignment for FP8 (128-bit alignment)
        if (reinterpret_cast<uintptr_t>(args.ref_A.data()) % 16 != 0 ||
            reinterpret_cast<uintptr_t>(args.ref_B.data()) % 16 != 0 ||
            reinterpret_cast<uintptr_t>(args.ref_D.data()) % 16 != 0) {
            return false;
        }

        // Validate FP8 scaling factors (reusing existing alpha and output_scale)
        if (args.alpha <= 0.0f || args.output_scale <= 0.0f) {
            return false;
        }

        return true;
    }

    static size_t get_workspace_size(Arguments const& args) {
        return SingleGemm::get_workspace_size(args);
    }

    static cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        SingleGemm single_gemm_op;
        return single_gemm_op.run(args, workspace, stream);
    }
};

// Specialization helper for FP8 E4M3 with default tile sizes optimized for L4
template <typename ElementType, typename AccumElementType>
using DeviceGemmGatedSm89SingleFP8Default = DeviceGemmGatedSm89SingleFP8<
    ElementType,
    AccumElementType,
    cutlass::gemm::GemmShape<128, 128, 32>,  // CTA shape optimized for FP8: deeper K for 16x8x32
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp shape optimized for FP8
    cutlass::gemm::GemmShape<1, 1, 1>        // Cluster shape for SM89
>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
