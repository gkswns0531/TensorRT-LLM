/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

// CUTLASS 2.x includes for SM89
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/activation.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// SM89 Fused Gated GEMM kernel template (L4 GPU)
template <typename ElementType, typename AccumElementType, typename CTAShape, typename WarpShape_, typename ClusterShape,
    template <class /* ElementCompute */> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = false>
struct DeviceGemmGatedSm89
{
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>, 
                  "SM89 supports FP16/BF16 for fused gated activation");

    // A matrix configuration
    using ElementA = ElementType;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    // B matrix configuration  
    using ElementB = ElementType;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    // C/D matrix configuration
    using ElementC = ElementType;
    using LayoutC = typename std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementD = ElementType;
    using LayoutD = LayoutC;
    static constexpr int AlignmentD = AlignmentC;

    // Core tensor op type
    using ElementAccumulator = AccumElementType;
    using ElementCompute = AccumElementType;

    // CUTLASS 2.x compatible threadblock configuration
    using ThreadblockShape = CTAShape;
    using WarpShape = WarpShape_;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;  // SM89도 동일한 최적 instruction shape
    
    // SwiGLU Epilogue - 현재는 기본 SiLU 사용 (임시)
    // 실제 SwiGLU는 두 개의 GEMM이 필요하므로 단일 epilogue로 구현 불가
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementD, 128 / cutlass::sizeof_bits<ElementD>::value,
        ElementAccumulator, ElementCompute>;

    // SM89 최적화된 CUTLASS 2.x device::Gemm
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,  
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,  // SM89 architecture
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,  // 4 stages - SM89도 4+ stages 최적화
        AlignmentA,
        AlignmentB,
        false,  // SplitKSerial
        cutlass::arch::OpMultiplyAdd>;  // Operator

    using Arguments = typename Gemm::Arguments;
    // Note: CUTLASS 2.x uses Arguments, not Params

    static cutlass::Status can_implement(Arguments const& args) {
        Gemm gemm_op;
        return gemm_op.can_implement(args);
    }
    
    static size_t get_workspace_size(Arguments const& args) {
        return Gemm::get_workspace_size(args);
    }
    
    static cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        Gemm gemm_op;
        
        // CUTLASS 2.x device::Gemm pattern: initialize then run
        cutlass::Status status = gemm_op.initialize(args, workspace);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }
        
        return gemm_op.run(stream);
    }
};

// SM89 최적화된 설정 (SM80과 동일한 최적화 적용)
template <typename ElementType>
struct Sm89GatedGemmConfigs {
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>);

    // 문서 권장: 중간 크기 문제에 최적화된 CTA shape
    using DefaultCTAShape = cutlass::gemm::GemmShape<128, 256, 32>;
    // 문서 권장: Tensor Core 효율성을 위한 warp shape
    using DefaultWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using DefaultClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    // SM89 최적 InstructionShape
    using DefaultInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    
    // 4-stage 파이프라이닝 (문서 권장)
    static constexpr int DefaultStages = 4;
    
    template<class T> using DefaultActivation = cutlass::epilogue::thread::SiLu<T>;
};

template<typename ElementType>
using DefaultDeviceGemmGatedSm89 = DeviceGemmGatedSm89<
    ElementType, float,  // AccumElementType = float
    typename Sm89GatedGemmConfigs<ElementType>::DefaultCTAShape,
    typename Sm89GatedGemmConfigs<ElementType>::DefaultWarpShape,  // WarpShape 추가
    typename Sm89GatedGemmConfigs<ElementType>::DefaultClusterShape,
    Sm89GatedGemmConfigs<ElementType>::template DefaultActivation
>;

} // namespace cutlass_kernels
} // namespace kernels  
} // namespace tensorrt_llm