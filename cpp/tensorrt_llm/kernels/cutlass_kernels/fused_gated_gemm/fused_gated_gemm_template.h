/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC

#include "fused_gated_gemm.h"
#include "fused_gated_gemm_kernel_template_sm80.h"
#include "fused_gated_gemm_kernel_template_sm89.h"
#include "fused_gated_gemm_kernel_template_sm90.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#include <algorithm>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

template <typename Gemm>
size_t typedGemmGatedKernelLauncher(Gemm gemm, typename Gemm::Arguments args, void* D, void const* A, void const* B,
    void const* C_bias, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementT = typename Gemm::ElementA;

    // Check shared memory size; throw when SMEM exceeds
    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
    static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();
    if (smem_size > mMaxSmemSize)
    {
        std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got "
            + std::to_string(mMaxSmemSize);
        throw std::runtime_error("[TensorRT-LLM Error][fusedGatedGemm Runner] " + errMsg);
    }

    // Return workspace size
    if (!A && !B && !C_bias && !D)
    {
        return gemm.get_workspace_size(args);
    }

    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        std::string errMsg("Requested workspace size insufficient. Required "
            + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));
        throw std::runtime_error("[TensorRT-LLM Error][fusedGatedGemm Runner] " + errMsg);
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "fusedGatedGemm cutlass kernel not implemented given the params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT-LLM Error][fusedGatedGemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to initialize. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[TensorRT-LLM Error][fusedGatedGemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to run gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[TensorRT-LLM Error][fusedGatedGemm Runner] " + errMsg);
    }
    return gemm.get_workspace_size(args);
}

template <typename Gemm, bool SwapAB>
typename Gemm::Arguments prepareGemmArgsSm90(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig)
{
    using ElementT = typename Gemm::ElementA;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    int arg_m = m;
    int arg_n = n / 2;
    ElementT const* ptr_A = reinterpret_cast<ElementT const*>(A);
    ElementT const* ptr_B = reinterpret_cast<ElementT const*>(B);
    if constexpr (SwapAB)
    {
        arg_m = n / 2;
        arg_n = m;
        ptr_A = reinterpret_cast<ElementT const*>(B);
        ptr_B = reinterpret_cast<ElementT const*>(A);
    }
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(arg_m, k, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(arg_n, k, 1));
    StrideC stride_C;
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(arg_m, arg_n, 1));
    typename Gemm::Arguments args = {cutlass::gemm::GemmUniversalMode::kGemm, {arg_m, arg_n, k, 1},
        {ptr_A, stride_A, ptr_B, stride_B, scale_d0, scale_d1},
        {{}, // epilogue.thread
            nullptr, stride_C, reinterpret_cast<ElementT*>(D), stride_D}};
    args.epilogue.thread.alpha = scale_output;
    return args;
}

template <typename T, typename CTAShape, typename ClusterShape,
    template <class> typename Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = true>
size_t genericGemmGatedKernelLauncherSm90(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using ElementT = typename TllmToCutlassTypeAdapter<T>::type;
    using AccumElementType = float;
    
    // FP8 E4M3 specialized TMA schedules with fast accumulation
    using MainloopScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
        cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
    using EpilogueScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
        cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;
    using TileSchedulerType = void;
    using Gemm = typename DeviceGemmGatedSm90<ElementT, AccumElementType, CTAShape, ClusterShape, MainloopScheduleType,
        EpilogueScheduleType, TileSchedulerType, Activation, SwapAB>::Gemm;
    auto args = prepareGemmArgsSm90<Gemm, SwapAB>(
        D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, scale_output, gemmConfig);
    return typedGemmGatedKernelLauncher(Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
#else  // COMPILE_HOPPER_TMA_GEMMS
    throw std::runtime_error(
        "[TensorRT-LLm Error][GemmGatedKernelLauncherSm90] Please recompile with support for hopper by passing 90-real "
        "as an arch to build_wheel.py.");
#endif // COMPILE_HOPPER_TMA_GEMMS
}

// SM80 config dispatch function using CUTLASS 2.x device::Gemm
template <typename T, int CtaM, int CtaN, int CtaK, int WarpM, int WarpN, int WarpK>
size_t dispatchGemmConfigSm80(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // Convert TensorRT-LLM types to CUTLASS types
    using ElementType = typename std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, cutlass::bfloat16_t>;
    using AccumElementType = float;
    
    // Define CTA and Warp shapes 
    using CTAShape = cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // SM80 표준 InstructionShape (DefaultGemmConfiguration 호환)
    
    // Use SM80 device GEMM with 수정된 template signature (Activation parameter 제거)
    using DeviceKernel = DeviceGemmGatedSm80<ElementType, AccumElementType, CTAShape, WarpShape,
        cutlass::gemm::GemmShape<1, 1, 1>>;  // ClusterShape (SwapAB는 기본값 false 사용)
    
    // 진정한 SwiGLU를 위한 DualGemm 텐서 구성
    cutlass::TensorRef<ElementType const, cutlass::layout::RowMajor> tensor_a(
        reinterpret_cast<ElementType const*>(A), cutlass::layout::RowMajor::packed({m, k}));
    
    // B 매트릭스: DualGemm이 내부에서 [B_linear | B_gate]로 분할 처리  
    cutlass::TensorRef<ElementType const, cutlass::layout::ColumnMajor> tensor_b(
        reinterpret_cast<ElementType const*>(B), cutlass::layout::ColumnMajor::packed({k, n}));
    
    cutlass::TensorRef<ElementType const, cutlass::layout::RowMajor> tensor_c(
        reinterpret_cast<ElementType const*>(C_bias), cutlass::layout::RowMajor::packed({1, n}));
    
    // 최종 출력: n/2 크기 (SwiGLU 결과)
    cutlass::TensorRef<ElementType, cutlass::layout::RowMajor> tensor_d(
        reinterpret_cast<ElementType*>(D), cutlass::layout::RowMajor::packed({m, n/2}));

    // DualGemm Arguments: B 매트릭스를 통째로 전달하고 내부에서 분할 처리
    typename DeviceKernel::Arguments arguments(
        {m, n/2, k},                    // problem_size (SwiGLU 출력 크기)
        tensor_a,                       // A 매트릭스
        tensor_b,                       // B 매트릭스 전체 (내부에서 linear/gate 분할)
        tensor_c,                       // C bias
        tensor_d,                       // D 출력
        scale_d0,                       // alpha
        scale_d1                        // beta
    );
    
    DeviceKernel gemm_operator;
    
    // Check if the operation is supported
    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm80] Cannot implement GEMM with given arguments. Status: " 
                                + std::to_string(static_cast<int>(status));
        throw std::runtime_error(error_msg);
    }
    
    // Get workspace size
    size_t workspace_size = gemm_operator.get_workspace_size(arguments);
    if (workspace_size > workspaceBytes) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm80] Insufficient workspace. Required: " 
                                + std::to_string(workspace_size) + ", Available: " + std::to_string(workspaceBytes);
        throw std::runtime_error(error_msg);
    }
    
    // DualGemm execution: 내부에서 dual GEMM + SwiGLU 융합 실행
    status = gemm_operator.run(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm80] Kernel execution failed. Status: " 
                                + std::to_string(static_cast<int>(status));
        throw std::runtime_error(error_msg);
    }
    
    return workspace_size;
}

template <typename T, typename CTAShape>
size_t dispatchGemmConfigSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_2, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _2, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_2, _2, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x8x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _8, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_8x1x1:
        return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_8, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner][dispatchGemmConfigSm90] Config is invalid for fused "
            "gated GEMM.");
        break;
    }
}

// Forward declarations for architecture-specific dispatch functions
template <typename T>
size_t dispatchGemmToCutlassSm80(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

template <typename T>
size_t dispatchGemmToCutlassSm89(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

template <typename T>
size_t dispatchGemmToCutlassSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

// SM80 dispatch function for FP16/BF16 using CUTLASS 2.x device::Gemm 
template <typename T>
size_t dispatchGemmToCutlassSm80(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // Support FP16 and BF16 data types
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>, 
                  "dispatchGemmToCutlassSm80 supports FP16 and BF16 only");
    
    switch (gemmConfig.tile_config_sm80)
    {
    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        return dispatchGemmConfigSm80<T, 64, 128, 64, 32, 64, 64>(D, A, B, C_bias, quantOption, m, n, k, 
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm80<T, 128, 128, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        return dispatchGemmConfigSm80<T, 128, 256, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm80<T, 256, 128, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        return dispatchGemmConfigSm80<T, 128, 64, 64, 64, 32, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        return dispatchGemmConfigSm80<T, 64, 64, 128, 32, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x64x128_WarpShape64x32x128:
        return dispatchGemmConfigSm80<T, 128, 64, 128, 64, 32, 128>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    default:
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmToCutlassSm80] Config undefined for tile config: " + std::to_string(static_cast<int>(gemmConfig.tile_config_sm80));
        throw std::runtime_error(error_msg);
    }

    return 0;
}

// SM89 config dispatch function for FP16/BF16 using CUTLASS 2.x device::Gemm
template <typename T, int CtaM, int CtaN, int CtaK, int WarpM, int WarpN, int WarpK>
size_t dispatchGemmConfigSm89(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // Convert TensorRT-LLM types to CUTLASS types
    using ElementType = typename std::conditional_t<std::is_same_v<T, half>, cutlass::half_t, cutlass::bfloat16_t>;
    using AccumElementType = float;
    
    // Define CTA and Warp shapes 
    using CTAShape = cutlass::gemm::GemmShape<CtaM, CtaN, CtaK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // SM89 표준 InstructionShape (DefaultGemmConfiguration 호환)
    
    // Use SM89-dedicated device GEMM with 수정된 template signature (Activation parameter 제거)
    using DeviceKernel = DeviceGemmGatedSm89<ElementType, AccumElementType, CTAShape, WarpShape, 
        cutlass::gemm::GemmShape<1, 1, 1>>;  // ClusterShape (SwapAB는 기본값 false 사용)
    
    // 진정한 SwiGLU를 위한 DualGemm 텐서 구성 (SM89 = L4)
    cutlass::TensorRef<ElementType const, cutlass::layout::RowMajor> tensor_a(
        reinterpret_cast<ElementType const*>(A), cutlass::layout::RowMajor::packed({m, k}));
    
    // B 매트릭스: DualGemm이 내부에서 [B_linear | B_gate]로 분할 처리  
    cutlass::TensorRef<ElementType const, cutlass::layout::ColumnMajor> tensor_b(
        reinterpret_cast<ElementType const*>(B), cutlass::layout::ColumnMajor::packed({k, n}));
    
    cutlass::TensorRef<ElementType const, cutlass::layout::RowMajor> tensor_c(
        reinterpret_cast<ElementType const*>(C_bias), cutlass::layout::RowMajor::packed({1, n}));
    
    // 최종 출력: n/2 크기 (SwiGLU 결과)
    cutlass::TensorRef<ElementType, cutlass::layout::RowMajor> tensor_d(
        reinterpret_cast<ElementType*>(D), cutlass::layout::RowMajor::packed({m, n/2}));

    // DualGemm Arguments: B 매트릭스를 통째로 전달하고 내부에서 분할 처리
    typename DeviceKernel::Arguments arguments(
        {m, n/2, k},                    // problem_size (SwiGLU 출력 크기)
        tensor_a,                       // A 매트릭스
        tensor_b,                       // B 매트릭스 전체 (내부에서 linear/gate 분할)
        tensor_c,                       // C bias
        tensor_d,                       // D 출력
        scale_d0,                       // alpha
        scale_d1                        // beta
    );
    
    DeviceKernel gemm_operator;
    
    // Check if the operation is supported
    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm89] Cannot implement GEMM with given arguments. Status: " 
                                + std::to_string(static_cast<int>(status));
        throw std::runtime_error(error_msg);
    }
    
    // Get workspace size
    size_t workspace_size = gemm_operator.get_workspace_size(arguments);
    if (workspace_size > workspaceBytes) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm89] Insufficient workspace. Required: " 
                                + std::to_string(workspace_size) + ", Available: " + std::to_string(workspaceBytes);
        throw std::runtime_error(error_msg);
    }
    
    // CUTLASS 2.x device::Gemm execution pattern: initialize + run  
    // DualGemm execution: 내부에서 dual GEMM + SwiGLU 융합 실행
    status = gemm_operator.run(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmConfigSm89] Kernel execution failed. Status: " 
                                + std::to_string(static_cast<int>(status));
        throw std::runtime_error(error_msg);
    }
    
    return workspace_size;
}

// SM89 dispatch function for FP16/BF16 using CUTLASS 2.x 
template <typename T>
size_t dispatchGemmToCutlassSm89(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // Support FP16 and BF16 data types on SM89
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>, 
                  "dispatchGemmToCutlassSm89 supports FP16 and BF16 only");
    
    // Follow fp8_rowwise_gemm pattern: dispatch to different tile configs
    switch (gemmConfig.tile_config_sm80)
    {
    case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        return dispatchGemmConfigSm89<T, 32, 128, 64, 32, 32, 64>(D, A, B, C_bias, quantOption, m, n, k, 
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        return dispatchGemmConfigSm89<T, 64, 128, 64, 32, 64, 64>(D, A, B, C_bias, quantOption, m, n, k, 
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, 128, 128, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, 128, 256, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        return dispatchGemmConfigSm89<T, 256, 128, 64, 64, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        return dispatchGemmConfigSm89<T, 128, 64, 64, 64, 32, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        return dispatchGemmConfigSm89<T, 64, 64, 128, 32, 64, 64>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x64x128_WarpShape64x32x128:
        return dispatchGemmConfigSm89<T, 128, 64, 128, 64, 32, 128>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    default:
        std::string error_msg = "[TensorRT-LLM Error][dispatchGemmToCutlassSm89] Config undefined for tile config: " + std::to_string(static_cast<int>(gemmConfig.tile_config_sm80));
        throw std::runtime_error(error_msg);
    }

    return 0;
}

template <typename T>
size_t dispatchGemmToCutlassSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // SM90 TMA currently supports FP8 E4M3 only
    // FP16/BF16 support will be added in future PR with low_latency_gemm integration
    static_assert(std::is_same_v<T, __nv_fp8_e4m3>, "fusedGatedGemmSm90 only supports FP8(e4m3)");
    constexpr int Ktile = 128 / sizeof(T);
    using _Ktile = Int<Ktile>;
    switch (gemmConfig.tile_config_sm90)
    {
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
        return dispatchGemmConfigSm90<T, Shape<_64, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
        return dispatchGemmConfigSm90<T, Shape<_128, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
            scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM90::Undefined:
        throw std::runtime_error(
            "[TensorRT-LLm Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] gemm config undefined.");
        break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT-LLm Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] gemm config should have "
            "already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLm Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] Config is invalid for fused "
            "gated GEMM.");
        break;
    }
}

template <typename T>
CutlassFusedGatedGemmRunner<T>::CutlassFusedGatedGemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mSm = tk::getSMVersion();
    
    // Initialize multiProcessorCount following TensorRT-LLM pattern
    auto const deviceId = tk::getDevice();
    cudaDeviceProp deviceProp{};
    tk::check_cuda_error(cudaGetDeviceProperties(&deviceProp, deviceId));
    mMultiProcessorCount = deviceProp.multiProcessorCount;
}

template <typename T>
CutlassFusedGatedGemmRunner<T>::~CutlassFusedGatedGemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::dispatchToArch(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // Architecture-based dispatch following TensorRT-LLM patterns
    if (mSm >= 90)
    {
        // SM90+ (H100, B100+) -> FP8 E4M3 uses CUTLASS 3.x with TMA + WGMMA
        // FP16/BF16 will be implemented in future PR using low_latency_gemm 
        return dispatchGemmToCutlassSm90<T>(D, A, B, C_bias, quantOption, m, n, k, 
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
    else if (mSm == 89)
    {
        // SM89 (L4) -> FP16/BF16 SwiGLU fusion fully implemented with dedicated SM89 CUTLASS 2.x kernels
        return dispatchGemmToCutlassSm89<T>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
    else if (mSm >= 80)
    {
        // SM80 (A100) -> FP16/BF16 SwiGLU fusion fully implemented with CUTLASS 2.x device::Gemm
        return dispatchGemmToCutlassSm80<T>(D, A, B, C_bias, quantOption, m, n, k,
            scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
    else
    {
        std::string error_msg = "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS fused gated GEMM. Requires SM80+ (A100+). Current: SM" 
                                + std::to_string(mSm) + " with " + typeid(T).name();
        throw std::runtime_error(error_msg);
    }
}

template <typename T>
void CutlassFusedGatedGemmRunner<T>::gemm(void* D, void const* A, void const* B, void const* C_bias,
    tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatchToArch(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, scale_output, gemmConfig, workspace,
        workspaceBytes, stream, occupancy);
}

template <typename T>
std::vector<tkc::CutlassGemmConfig> CutlassFusedGatedGemmRunner<T>::getConfigs() const
{
    using tkc::CutlassTileConfig;
    using tkc::CutlassGemmConfig;
    using tkc::SplitKStyle;

    std::vector<CutlassGemmConfig> candidateConfigs;

    // Current implementation status:
    // - A100/L4 (SM80/89): FP16/BF16 SwiGLU fusion fully implemented with CUTLASS 2.x
    // - H100 (SM90): FP8 E4M3 supported, FP16/BF16 will be implemented in future PR
    if constexpr (std::is_same_v<T, half>)
    {
        if (mSm < 80)
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] FP16 fused gated GEMM requires SM80+ (A100+). Current: SM" + std::to_string(mSm));
        }
        
        // FP16 SwiGLU fusion - A100/L4 support only
        // H100 FP16 will be implemented in future PR with low_latency_gemm integration  
        if (mSm >= 90)
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] H100 FP16 SwiGLU fusion will be implemented "
                "in future release with advanced TMA optimization. Current implementation supports A100/L4 only.");
        }
        
        // SM80/89: CUTLASS 2.x kernels with FP16 SwiGLU-optimized tile configurations
        auto config_type_param = tkc::CutlassGemmConfig::CandidateConfigTypeParam::FP16_SWIGLU;
        std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 1, config_type_param);
        candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        if (mSm < 80)
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] BF16 fused gated GEMM requires SM80+ (A100+). Current: SM" + std::to_string(mSm));
        }
        
        // BF16 SwiGLU fusion - A100/L4 support only
        // H100 BF16 will be implemented in future PR with low_latency_gemm integration
        if (mSm >= 90)
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] H100 BF16 SwiGLU fusion will be implemented "
                "in future release with advanced TMA optimization. Current implementation supports A100/L4 only.");
        }
        
        // SM80/89: CUTLASS 2.x kernels with BF16 SwiGLU-optimized tile configurations  
        auto config_type_param = tkc::CutlassGemmConfig::CandidateConfigTypeParam::FP16_SWIGLU;
        std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 1, config_type_param);
        candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
    }
    else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        if (mSm < 89)
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] FP8 E4M3 fused gated GEMM requires SM89+ (L4+). Current: SM" + std::to_string(mSm));
        }
        
        if (mSm == 90)
        {
            // SM90 uses HOPPER configs with TMA
            tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param
                = tkc::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;
            std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 2, config_type_param);
            candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
            
            // registers are not enough when N_tile is 256, remove some configs
            candidateConfigs.erase(std::remove_if(candidateConfigs.begin(), candidateConfigs.end(),
                                       [](auto const& config)
                                       {
                                           return config.tile_config_sm90 == tkc::CutlassTileConfigSM90::CtaShape64x256x128B
                                               || config.tile_config_sm90
                                               == tkc::CutlassTileConfigSM90::CtaShape128x256x128B;
                                       }),
                candidateConfigs.end());
            
            std::vector<tkc::CutlassTileConfigSM90> tilesSm90
                = {tkc::CutlassTileConfigSM90::CtaShape64x16x128B, tkc::CutlassTileConfigSM90::CtaShape64x32x128B,
                    tkc::CutlassTileConfigSM90::CtaShape64x64x128B, tkc::CutlassTileConfigSM90::CtaShape64x128x128B,
                    tkc::CutlassTileConfigSM90::CtaShape128x16x128B, tkc::CutlassTileConfigSM90::CtaShape128x32x128B,
                    tkc::CutlassTileConfigSM90::CtaShape128x64x128B, tkc::CutlassTileConfigSM90::CtaShape128x128x128B};
            for (auto const& tile_config : tilesSm90)
            {
                {
                    CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                        tkc::ClusterShape::ClusterShape_1x8x1);
                    candidateConfigs.push_back(config);
                }
                {
                    CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                        tkc::ClusterShape::ClusterShape_8x1x1);
                    candidateConfigs.push_back(config);
                }
            }
        }
        else
        {
            // SM89 FP8: Not yet implemented - will be supported with dedicated CUTLASS 2.x kernel
            throw std::runtime_error(
                "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] SM89 FP8 SwiGLU fusion not yet implemented. "
                "Will be supported with dedicated CUTLASS 2.x kernel in future releases.");
        }
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLM Error][CutlassFusedGatedGemmRunner] Unsupported data type for fused gated GEMM: " + std::string(typeid(T).name()));
    }
    
    return candidateConfigs;
}

// Note: can be quite heavyweight; when possible, call once
template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t workspace_size = 0;
    auto gemmConfigs = CutlassFusedGatedGemmRunner<T>{}.getConfigs();
    for (auto const& gemmConfig : gemmConfigs)
    {
        try
        {
            size_t curr_workspace_size = CutlassFusedGatedGemmRunner<T>::dispatchToArch(
                nullptr, nullptr, nullptr, nullptr, tk::QuantMode{}, m, n, k, 1.0, 1.0, 1.0, gemmConfig, nullptr, 0, 0);
            workspace_size = std::max(workspace_size, curr_workspace_size);
        }
        catch (std::runtime_error& e)
        {
            // Swallow errors when SMEM exceeds maximum allowed
            continue;
        }
    }

    return workspace_size;
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Custom hash function for the MNK type
    using MNK = std::tuple<int, int, int>;

    struct MNKHash
    {
        size_t operator()(const MNK& mnk) const
        {
            auto h1 = std::hash<int>{}(std::get<0>(mnk));
            auto h2 = std::hash<int>{}(std::get<1>(mnk));
            auto h3 = std::hash<int>{}(std::get<2>(mnk));
            return h1 ^ h2 ^ h3;
        }
    };

    static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

    size_t workspace_size = 0;
    if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end())
    {
        workspace_size = CutlassFusedGatedGemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
        workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
    }
    else
    {
        workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
    }
    return workspace_size;
}



// Follow int8_gemm pattern: single template function for all architectures
template <typename T, typename arch>
size_t dispatchGemmToCutlass(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
    int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
    char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    if constexpr (std::is_same_v<arch, cutlass::arch::Sm90>)
    {
        // SM90 uses existing CUTLASS 3.x + TMA implementation
        if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
        {
            // FP8 E4M3 - fully implemented in main branch
            return dispatchGemmToCutlassSm90<T>(D, A, B, C_bias, quantOption, m, n, k, 
                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>)
        {
            // H100 FP16/BF16 optimization will be implemented in future PR using low_latency_gemm
            // Current PR focuses on A100/L4 FP16/BF16 support only
            throw std::runtime_error(
                "[TensorRT-LLM Error][dispatchGemmToCutlass] H100 FP16/BF16 SwiGLU fusion will be implemented "
                "in future release with advanced TMA optimization. Current implementation supports A100/L4 only.");
        }
        else
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][dispatchGemmToCutlass] SM90 unsupported data type: " + std::string(typeid(T).name()));
        }
    }
    else if constexpr (std::is_same_v<arch, cutlass::arch::Sm89>)
    {
        // SM89 (L4) - supports FP16, BF16 via CUTLASS 2.x (following fp8_rowwise_gemm pattern)
        if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
        {
            // SM89 FP8 support will be implemented in future with dedicated CUTLASS 2.x kernel
            throw std::runtime_error(
                "[TensorRT-LLM Error][dispatchGemmToCutlass] SM89 FP8 SwiGLU fusion not yet implemented. "
                "Will be supported with dedicated CUTLASS 2.x kernel in future releases.");
        }
        else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>)
        {
            // FP16/BF16 on SM89 - use dedicated SM89 CUTLASS 2.x kernels
            return dispatchGemmToCutlassSm89<T>(D, A, B, C_bias, quantOption, m, n, k,
                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else
        {
            throw std::runtime_error(
                "[TensorRT-LLM Error][dispatchGemmToCutlass] SM89 unsupported data type: " + std::string(typeid(T).name()));
        }
    }
    else
    {
        // SM80 uses CUTLASS 2.x device::Gemm with SM80 kernels
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>)
        {
            return dispatchGemmToCutlassSm80<T>(D, A, B, C_bias, quantOption, m, n, k,
                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else
        {
            // FP8 on SM80 not supported - requires SM89+ (L4+)  
            throw std::runtime_error(
                "[TensorRT-LLM Error][dispatchGemmToCutlass] FP8 requires SM89+ (L4+) with SM90 kernels. Current arch: " + std::string(typeid(arch).name()));
        }
    }
    
    return 0;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
