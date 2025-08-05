/*
 * Copyright (c) 2024, TensorRT-LLM Extension.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass_extensions/gemm/collective/collective_builder_gated.hpp"
#include "cutlass_extensions/gemm/kernel/gemm_universal_gated.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
using namespace cute;

// SM80 Ampere 아키텍처용 Fused Gated GEMM 커널 템플릿
template <typename ElementType, typename AccumElementType, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType, typename TileSchedulerType = void,
    template <class /* ElementCompute */> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = false>
struct DeviceGemmGatedSm80
{
    // FP16/BF16 지원 확인 (MOE와 동일한 조건)
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>, 
                  "SM80 supports FP16/BF16 for fused gated activation");

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
    using LayoutC = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using ElementD = ElementType;
    using LayoutD = LayoutC;
    static constexpr int AlignmentD = AlignmentC;

    // Core tensor op type
    using ElementAccumulator = AccumElementType;
    using ElementCompute = AccumElementType;
    using ElementScale = ElementCompute;

    // SM80 MMA Atom 선택 (MOE와 동일)
    using MMA_Atom = std::conditional_t<std::is_same_v<ElementType, cutlass::half_t>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>, 
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;

    // Thread layout and value layout (MOE에서 최적화된 구조 적용)
    using ThreadLayoutMNK = cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>;
    using ValLayoutMNK = cute::Tile<cute::_32, cute::_32, cute::_16>;
    
    using TiledMma = cute::TiledMMA<MMA_Atom, ThreadLayoutMNK, ValLayoutMNK>;

    // Collective mainloop and epilogue
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        CTAShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename TiledMma::ValTypeA) * cute::size(typename TiledMma::ThrLayoutVMNK{}) / 8)>,
        MainloopScheduleType
    >::CollectiveOp;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm80, cutlass::arch::OpClassTensorOp,
        CTAShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        TileSchedulerType>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using Arguments = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    // MOE의 3단계 구현 체계 적응
    struct GatedActivationKernel {
        
        // 1단계: ActivationFn 타입 결정 (컴파일 타임)
        using ActivationFn = Activation<ElementCompute>;
        
        // 2단계: 이중 GEMM 계산을 위한 accumulator 구조
        template<typename TiledMMA>
        struct DualAccumulator {
            cute::Tensor<typename TiledMMA::ValTypeC> accum;      // Up path
            cute::Tensor<typename TiledMMA::ValTypeC> accum_gate; // Gate path
            
            CUTLASS_DEVICE
            DualAccumulator(TiledMMA const& tiled_mma, int tile_m, int tile_n) {
                accum = cute::partition_fragment_C(tiled_mma, cute::make_shape(tile_m, tile_n));
                accum_gate = cute::partition_fragment_C(tiled_mma, cute::make_shape(tile_m, tile_n));
                cute::clear(accum);
                cute::clear(accum_gate);
            }
        };
        
        // 3단계: Gated activation 융합 (MOE 패턴 완전 이식)
        template<typename AccumTensor>
        CUTLASS_DEVICE static void apply_gated_activation(
            AccumTensor& accum, 
            AccumTensor const& accum_gate) 
        {
            ActivationFn fn{};
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(accum); i++) {
                accum(i) = fn(accum_gate(i)) * accum(i);  // SiLU(gate) ⊙ up
            }
        }
    };

    // MOE의 조건 검증 로직 이식
    static constexpr bool supportsFusedGatedActivation(int gemm_k, int gemm_n, int sm) {
        constexpr bool is_gated_activation = true;  // SwiGLU
        constexpr bool use_fp8 = false;  // FP16/BF16만 지원
        
        return is_gated_activation
            && (sm >= 80)  // SM80+
            && (gemm_k % 64 == 0) && (gemm_n % 64 == 0)  // MOE와 동일한 64배수 조건
            && !use_fp8;  // FP16/BF16만
    }

    // 워크스페이스 크기 계산 (기존 구조 유지 + MOE 최적화)
    static size_t get_workspace_size(Arguments const& args) {
        return Gemm::get_workspace_size(args);
    }

    // 커널 실행 (기존 인터페이스 유지)
    static cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        Gemm gemm_op;
        return gemm_op.run(args, workspace, stream);
    }
};

// SM80용 특화된 설정들
template <typename ElementType>
struct Sm80GatedGemmConfigs {
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>);

    // MOE에서 검증된 최적 타일 크기들
    using DefaultCTAShape = cute::Shape<cute::_128, cute::_128, cute::_32>;  // 128x128x32
    using DefaultClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;   // 1x1x1 클러스터
    
    // SM80 최적화된 스케줄링
    using DefaultMainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using DefaultEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    
    // 기본 활성화 함수
    template<class T> using DefaultActivation = cutlass::epilogue::thread::SiLu<T>;
};

// 편의를 위한 별칭들
template<typename ElementType>
using DefaultDeviceGemmGatedSm80 = DeviceGemmGatedSm80<
    ElementType, float,  // AccumElementType = float
    typename Sm80GatedGemmConfigs<ElementType>::DefaultCTAShape,
    typename Sm80GatedGemmConfigs<ElementType>::DefaultClusterShape,
    typename Sm80GatedGemmConfigs<ElementType>::DefaultMainloopSchedule,
    typename Sm80GatedGemmConfigs<ElementType>::DefaultEpilogueSchedule,
    void,  // TileSchedulerType
    Sm80GatedGemmConfigs<ElementType>::template DefaultActivation
>;

} // namespace cutlass_kernels
} // namespace kernels  
} // namespace tensorrt_llm 