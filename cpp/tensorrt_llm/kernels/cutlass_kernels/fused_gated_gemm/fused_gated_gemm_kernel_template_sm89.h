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

#include "dual_gemm_swiglu_sm80.h"

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
    
    // SwiGLU Epilogue - Phase 1: 단일 GEMM + SiLU 적용  
    // 진정한 SwiGLU = linear * SiLU(gate)는 dual GEMM이 필요하므로
    // 현재는 기본 LinearCombinationSilu 사용 (Phase 2에서 dual GEMM 구현 예정)
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
        ElementD, 128 / cutlass::sizeof_bits<ElementD>::value,
        ElementAccumulator, ElementCompute>;

    // 진정한 SwiGLU를 위한 Dual GEMM 구현 (SM89 = L4)
    using DualGemm = DualGemmSwiGLU<
        ElementA, ElementB, ElementC, ElementD,
        LayoutA, LayoutB, LayoutC, LayoutD,
        ElementAccumulator, 
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm89,  // SM89 architecture (L4)
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        4  // 4 stages - 문서 권장 최적화
    >;

    using Arguments = typename DualGemm::Arguments;

    static cutlass::Status can_implement(Arguments const& args) {
        // DualGemm의 두 GEMM이 모두 실행 가능한지 확인
        typename DualGemm::LinearGemm linear_gemm;
        typename DualGemm::GateGemm gate_gemm;
        
        // Linear GEMM arguments 생성
        typename DualGemm::LinearGemm::Arguments linear_args(
            args.problem_size,
            args.ref_A,
            args.ref_B_linear,
            args.ref_C,
            cutlass::TensorRef<ElementC, LayoutC>(),  // 임시
            args.linear_epilogue
        );
        
        // Gate GEMM arguments 생성  
        typename DualGemm::GateGemm::Arguments gate_args(
            args.problem_size,
            args.ref_A,
            args.ref_B_gate,
            cutlass::TensorRef<ElementC const, LayoutC>(),
            cutlass::TensorRef<ElementC, LayoutC>(),  // 임시
            args.gate_epilogue
        );
        
        return (linear_gemm.can_implement(linear_args) == cutlass::Status::kSuccess &&
                gate_gemm.can_implement(gate_args) == cutlass::Status::kSuccess) ?
               cutlass::Status::kSuccess : cutlass::Status::kErrorInvalidProblem;
    }
    
    static size_t get_workspace_size(Arguments const& args) {
        return DualGemm::get_workspace_size(args);
    }
    
    static cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        DualGemm dual_gemm_op;
        return dual_gemm_op.run(args, workspace, stream);
    }
};

// SM89 (L4) 최적화된 설정 - SMEM 한계 고려
template <typename ElementType>
struct Sm89GatedGemmConfigs {
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>);

    // L4 SMEM 한계(100KB) 고려 - SwiGLU는 B 매트릭스가 2배이므로 작은 타일 사용
    using DefaultCTAShape = cutlass::gemm::GemmShape<128, 128, 32>;  // 256→128로 축소
    // 문서 권장: Tensor Core 효율성을 위한 warp shape
    using DefaultWarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using DefaultClusterShape = cutlass::gemm::GemmShape<1, 1, 1>;
    
    // SM89 최적 InstructionShape
    using DefaultInstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
    
    // 4-stage 파이프라이닝 (문서 권장)
    static constexpr int DefaultStages = 4;
    
    template<class T> using DefaultActivation = cutlass::epilogue::thread::SiLu<T>;
    
    // L4 특화: 더 작은 타일 옵션들 (SMEM 96KB < 100KB 보장)
    using SmallCTAShape = cutlass::gemm::GemmShape<64, 128, 32>;     // 소형 문제용
    using SmallWarpShape = cutlass::gemm::GemmShape<32, 64, 32>;     // 소형 문제용
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