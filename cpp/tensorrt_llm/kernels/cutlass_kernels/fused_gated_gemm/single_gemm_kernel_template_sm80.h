#pragma once

#include "single_gemm_swiglu_sm80.h"
#include "cutlass/gemm/gemm.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// DeviceGemmGatedSm80 wrapper for Single GEMM SwiGLU - replaces dual GEMM approach
template <typename ElementType, typename AccumElementType, typename CTAShape, typename WarpShape_, typename ClusterShape,
          bool SwapAB = false>
struct DeviceGemmGatedSm80Single
{
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
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // SM80 standard

    // Single GEMM SwiGLU operation
    using SingleGemm = SingleGemmSwiGLU<
        ElementA, ElementB, ElementC, ElementD,
        LayoutA, LayoutB, LayoutC, LayoutD,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        4  // Stages
    >;

    using Arguments = typename SingleGemm::Arguments;

    static bool can_implement(Arguments const& args) {
        // Check alignment requirements
        int m = args.problem_size.m();
        int n = args.problem_size.n();
        int k = args.problem_size.k();

        // Basic alignment checks
        if (k % 64 != 0 || n % 64 != 0) {
            return false;
        }

        // Check tensor alignment
        if (reinterpret_cast<uintptr_t>(args.ref_A.data()) % 16 != 0 ||
            reinterpret_cast<uintptr_t>(args.ref_B.data()) % 16 != 0 ||
            reinterpret_cast<uintptr_t>(args.ref_D.data()) % 16 != 0) {
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

// Specialization helper to match existing interface
template <typename ElementType, typename AccumElementType>
using DeviceGemmGatedSm80SingleDefault = DeviceGemmGatedSm80Single<
    ElementType,
    AccumElementType,
    cutlass::gemm::GemmShape<128, 128, 64>,  // Default CTA shape
    cutlass::gemm::GemmShape<64, 64, 64>,    // Default Warp shape  
    cutlass::gemm::GemmShape<1, 1, 1>        // Cluster shape for SM80
>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
