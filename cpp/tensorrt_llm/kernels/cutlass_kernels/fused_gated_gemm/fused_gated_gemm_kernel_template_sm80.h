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

// SM80 Fused Gated GEMM kernel template
template <typename ElementType, typename AccumElementType, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType, typename TileSchedulerType = void,
    template <class /* ElementCompute */> class Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = false>
struct DeviceGemmGatedSm80
{
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

    using MMA_Atom = std::conditional_t<std::is_same_v<ElementType, cutlass::half_t>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>, 
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;
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

    struct GatedActivationKernel {
        using ActivationFn = Activation<ElementCompute>;
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
        

        template<typename AccumTensor>
        CUTLASS_DEVICE static void apply_gated_activation(
            AccumTensor& accum, 
            AccumTensor const& accum_gate) 
        {
            ActivationFn fn{};
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(accum); i++) {
                accum(i) = fn(accum_gate(i)) * accum(i);
            }
        }
    };

    static constexpr bool supportsFusedGatedActivation(int gemm_k, int gemm_n, int sm) {
        constexpr bool is_gated_activation = true;
        constexpr bool use_fp8 = false;
        
        return is_gated_activation
            && (sm >= 80)
            && (gemm_k % 64 == 0) && (gemm_n % 64 == 0)
            && !use_fp8;
    }

    static size_t get_workspace_size(Arguments const& args) {
        return Gemm::get_workspace_size(args);
    }
    static cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        Gemm gemm_op;
        return gemm_op.run(args, workspace, stream);
    }
};

// SM80 specialized configurations
template <typename ElementType>
struct Sm80GatedGemmConfigs {
    static_assert(std::is_same_v<ElementType, cutlass::half_t> ||
                  std::is_same_v<ElementType, cutlass::bfloat16_t>);

    using DefaultCTAShape = cute::Shape<cute::_128, cute::_128, cute::_32>;
    using DefaultClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
    using DefaultMainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using DefaultEpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    
    template<class T> using DefaultActivation = cutlass::epilogue::thread::SiLu<T>;
};
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