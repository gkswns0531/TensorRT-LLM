/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "fused_gated_gemm_template.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

// Explicit template instantiation for all supported data types
// Following the unified architecture from Phase 1

// FP16 - Supported on SM80+ (A100+, L4+)
template class CutlassFusedGatedGemmRunner<half>;

// BF16 - Supported on SM80+ (A100+)  
#ifdef ENABLE_BF16
template class CutlassFusedGatedGemmRunner<__nv_bfloat16>;
#endif

// FP8 E4M3 - Supported on SM89+ (L4+)
template class CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>;

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm