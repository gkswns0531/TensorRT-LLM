#pragma once

#include "single_gemm_swiglu_sm89.h"
#include "cutlass/gemm/gemm.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// DeviceGemmGatedSm89 wrapper for Single GEMM SwiGLU - optimized for L4 (SM89)
template <typename ElementType, typename AccumElementType, typename CTAShape, typename WarpShape_, typename ClusterShape,
          bool SwapAB = false>
struct DeviceGemmGatedSm89Single
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
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;  // SM89 optimal instruction shape

    // Single GEMM SwiGLU operation for SM89 (L4)
    using SingleGemm = SingleGemmSwiGLUSm89<
        ElementA, ElementB, ElementC, ElementD,
        LayoutA, LayoutB, LayoutC, LayoutD,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,  // SM89 architecture tag for L4
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        4  // Stages - optimal for SM89
    >;

    using Arguments = typename SingleGemm::Arguments;

    static bool can_implement(Arguments const& args) {
        // Check alignment requirements for SM89
        int m = args.problem_size.m();
        int n = args.problem_size.n();
        int k = args.problem_size.k();

        // Basic alignment checks - SM89 requires 64-byte alignment
        if (k % 64 != 0 || n % 64 != 0) {
            return false;
        }

        // Check tensor alignment for SM89
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

// Specialization helper to match existing interface for SM89
template <typename ElementType, typename AccumElementType>
using DeviceGemmGatedSm89SingleDefault = DeviceGemmGatedSm89Single<
    ElementType,
    AccumElementType,
    cutlass::gemm::GemmShape<128, 128, 64>,  // Default CTA shape for SM89
    cutlass::gemm::GemmShape<64, 64, 64>,    // Default Warp shape for SM89  
    cutlass::gemm::GemmShape<1, 1, 1>        // Cluster shape for SM89
>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
