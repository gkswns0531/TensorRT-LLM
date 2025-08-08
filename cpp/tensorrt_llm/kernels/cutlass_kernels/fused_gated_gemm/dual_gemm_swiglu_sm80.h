#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

/**
 * @brief SM80/SM89용 진정한 SwiGLU 구현을 위한 Dual GEMM Wrapper
 * 
 * SM90의 CollectiveMmaGated 패턴을 CUTLASS 2.x로 포팅:
 * 1. Linear GEMM: A @ B_linear → linear_output  
 * 2. Gate GEMM:   A @ B_gate   → gate_output
 * 3. SwiGLU:      linear_output * SiLU(gate_output) → final_output
 */
template <typename ElementA, typename ElementB, typename ElementC, typename ElementD,
          typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD, 
          typename ElementAccumulator, typename OperatorClass, typename ArchTag, 
          typename ThreadblockShape, typename WarpShape, typename InstructionShape, 
          int Stages>
class DualGemmSwiGLU {
public:
    using ElementCompute = float;
    
    // 안전한 Alignment 설정 (DefaultGemmConfiguration 에러 방지)
    static constexpr int AlignmentA = 8;  // FP16/BF16에 대해 검증된 값
    static constexpr int AlignmentB = 8;
    static constexpr int AlignmentC = 8;
    
    // 안전한 Epilogue 정의 (Template template parameter 회피)
    using LinearEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, AlignmentC, ElementAccumulator, ElementCompute>;
    
    // Linear GEMM 설정: A @ B_linear (DefaultGemmConfiguration 호환)
    using LinearGemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        LinearEpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
        Stages,
        AlignmentA,
        AlignmentB>;
    
    // 안전한 Gate Epilogue 정의
    using GateEpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, AlignmentC, ElementAccumulator, ElementCompute>;
    
    // Gate GEMM 설정: A @ B_gate (DefaultGemmConfiguration 호환)
    using GateGemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        GateEpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        AlignmentA,
        AlignmentB>;

    // SwiGLU 융합 함수
    struct SwiGLUFusion {
        CUTLASS_HOST_DEVICE
        ElementD operator()(ElementC const& linear_val, ElementC const& gate_val) const {
            // SwiGLU = linear * SiLU(gate)
            // SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
            float gate_f = static_cast<float>(gate_val);
            float sigmoid_gate = gate_f / (1.0f + expf(-gate_f));  // SiLU
            float linear_f = static_cast<float>(linear_val);
            return static_cast<ElementD>(linear_f * sigmoid_gate);
        }
    };

    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;          // {m, n/2, k}
        cutlass::TensorRef<ElementA const, LayoutA> ref_A;
        cutlass::TensorRef<ElementB const, LayoutB> ref_B_linear;   // B의 첫 n/2 columns
        cutlass::TensorRef<ElementB const, LayoutB> ref_B_gate;     // B의 다음 n/2 columns  
        cutlass::TensorRef<ElementC const, LayoutC> ref_C;          // 전체 bias [b_u | b_g] (1 x 2n_out)
        cutlass::TensorRef<ElementD, LayoutD> ref_D;
        typename LinearGemm::EpilogueOutputOp::Params linear_epilogue;
        typename GateGemm::EpilogueOutputOp::Params gate_epilogue;
        ElementCompute output_scale;                                  // 최종 출력 스케일
        
        // 생성자
        Arguments(cutlass::gemm::GemmCoord const& problem_size,
                  cutlass::TensorRef<ElementA const, LayoutA> ref_A,
                  cutlass::TensorRef<ElementB const, LayoutB> ref_B,  // 전체 B
                  cutlass::TensorRef<ElementC const, LayoutC> ref_C,  // 전체 C (bias)
                  cutlass::TensorRef<ElementD, LayoutD> ref_D,
                  ElementCompute alpha = ElementCompute(1),
                  ElementCompute /*beta_unused*/ = ElementCompute(0),
                  ElementCompute output_scale = ElementCompute(1))
            : problem_size(problem_size)
            , ref_A(ref_A)
            , ref_C(ref_C)
            , ref_D(ref_D)
            , linear_epilogue({alpha, ElementCompute(0)})  // GEMM 단계에서는 bias 미적용 (beta=0)
            , gate_epilogue({alpha, ElementCompute(0)})    // GEMM 단계에서는 bias 미적용 (beta=0)
            , output_scale(output_scale)
        {
            // B 매트릭스 분할: [B_linear | B_gate]
            int n_out = problem_size.n();  // SwiGLU 출력 크기 (n/2)
            int k = problem_size.k();
            
            // 중요: dispatch에서 전달되는 B는 k × (2*n_out) 크기
            // 하지만 ref_B는 이미 올바른 레이아웃으로 구성됨
            
            // B_linear: 첫 번째 n_out columns (0 ~ n_out-1)
            ref_B_linear = cutlass::TensorRef<ElementB const, LayoutB>(
                ref_B.data(), 
                LayoutB::packed({k, n_out}));
            
            // B_gate: 두 번째 n_out columns (n_out ~ 2*n_out-1)
            // LayoutB가 ColumnMajor이면: k*n_out 오프셋
            // LayoutB가 RowMajor이면: n_out 오프셋
            ref_B_gate = cutlass::TensorRef<ElementB const, LayoutB>(
                ref_B.data() + (std::is_same_v<LayoutB, cutlass::layout::ColumnMajor> ? k * n_out : n_out),
                LayoutB::packed({k, n_out}));
        }
    };

    // 메모리 할당 함수
    static size_t get_workspace_size(Arguments const& args) {
        // 중간 결과 저장용 메모리: linear_output + gate_output
        int m = args.problem_size.m();
        int n = args.problem_size.n();
        size_t linear_bytes = sizeof(ElementC) * m * n;
        size_t gate_bytes = sizeof(ElementC) * m * n;
        return linear_bytes + gate_bytes;
    }

    // 실행 함수
    cutlass::Status run(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
        // 워크스페이스 설정
        if (!workspace) {
            return cutlass::Status::kErrorWorkspaceNull;
        }
        
        int m = args.problem_size.m();
        int n = args.problem_size.n();
        
        ElementC* linear_output = static_cast<ElementC*>(workspace);
        ElementC* gate_output = linear_output + m * n;
        
        // Linear output 텐서 설정
        cutlass::TensorRef<ElementC, LayoutC> linear_tensor(
            linear_output, cutlass::layout::RowMajor::packed({m, n}));
            
        // Gate output 텐서 설정  
        cutlass::TensorRef<ElementC, LayoutC> gate_tensor(
            gate_output, cutlass::layout::RowMajor::packed({m, n}));

        // 1. Linear GEMM 실행: A @ B_linear → linear_output
        LinearGemm linear_gemm;
        typename LinearGemm::Arguments linear_args(
            args.problem_size,
            args.ref_A,
            args.ref_B_linear,
            cutlass::TensorRef<ElementC const, LayoutC>(),  // beta=0, C 미사용
            linear_tensor,
            args.linear_epilogue
        );
        
        cutlass::Status status = linear_gemm.run(linear_args, stream);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }

        // 2. Gate GEMM 실행: A @ B_gate → gate_output
        GateGemm gate_gemm;
        typename GateGemm::Arguments gate_args(
            args.problem_size,
            args.ref_A,
            args.ref_B_gate,
            cutlass::TensorRef<ElementC const, LayoutC>(),  // C는 사용하지 않음
            gate_tensor,
            args.gate_epilogue
        );
        
        status = gate_gemm.run(gate_args, stream);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }

        // 3. SwiGLU 융합: (linear + b_u) * SiLU(gate + b_g) * output_scale → final_output
        // bias는 ref_C의 [0:n) = b_u, [n:2n) = b_g 로 가정(RowMajor 1 x 2n)
        ElementC const* bias_linear = args.ref_C.data();
        ElementC const* bias_gate = args.ref_C.data() ? (args.ref_C.data() + n) : nullptr;

        return launch_swiglu_fusion_impl(
            linear_output,
            gate_output,
            args.ref_D.data(),
            bias_linear,
            bias_gate,
            m,
            n,
            args.output_scale,
            stream);
    }

private:

    // SwiGLU 융합 커널 런처 (전역 함수 호출)
    template<typename ElementC_, typename ElementD_>
    cutlass::Status launch_swiglu_fusion_impl(
        ElementC_ const* linear_ptr,
        ElementC_ const* gate_ptr, 
        ElementD_* output_ptr,
        ElementC_ const* bias_linear_ptr,
        ElementC_ const* bias_gate_ptr,
        int m,
        int n,
        ElementCompute output_scale,
        cudaStream_t stream);
};

// SwiGLU 융합 커널 - 클래스 외부 전역 함수로 정의
template<typename ElementC, typename ElementD>
__global__ void swiglu_fusion_kernel(
    ElementC const* __restrict__ linear_ptr,
    ElementC const* __restrict__ gate_ptr, 
    ElementD* __restrict__ output_ptr,
    ElementC const* __restrict__ bias_linear_ptr,
    ElementC const* __restrict__ bias_gate_ptr,
    int m,
    int n,
    float output_scale) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = m * n;
    
    if (tid < total_elements) {
        ElementC linear_val = linear_ptr[tid];
        ElementC gate_val = gate_ptr[tid];
        
        // SwiGLU: (linear + b_u) * SiLU(gate + b_g) * output_scale
        int col = tid % n;
        float linear_f = static_cast<float>(linear_val);
        float gate_f = static_cast<float>(gate_val);

        if (bias_linear_ptr) {
            linear_f += static_cast<float>(bias_linear_ptr[col]);
        }
        if (bias_gate_ptr) {
            gate_f += static_cast<float>(bias_gate_ptr[col]);
        }

        float sigmoid_gate = gate_f / (1.0f + expf(-gate_f));  // SiLU
        float fused = (linear_f * sigmoid_gate) * output_scale;
        
        output_ptr[tid] = static_cast<ElementD>(fused);
    }
}

// 템플릿 특수화를 위한 구현부
template<typename ElementA, typename ElementB, typename ElementC, typename ElementD,
         typename LayoutA, typename LayoutB, typename LayoutC, typename LayoutD, 
         typename ElementAccumulator, typename OperatorClass, typename ArchTag, 
         typename ThreadblockShape, typename WarpShape, typename InstructionShape, 
         int Stages>
template<typename ElementC_, typename ElementD_>
cutlass::Status DualGemmSwiGLU<ElementA, ElementB, ElementC, ElementD, LayoutA, LayoutB, 
    LayoutC, LayoutD, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape, WarpShape, 
    InstructionShape, Stages>::launch_swiglu_fusion_impl(
        ElementC_ const* linear_ptr,
        ElementC_ const* gate_ptr,
        ElementD_* output_ptr,
        ElementC_ const* bias_linear_ptr,
        ElementC_ const* bias_gate_ptr,
        int m,
        int n,
        ElementCompute output_scale,
        cudaStream_t stream) {
    
    // GPU 커널 설정
    dim3 block(256);
    dim3 grid((m * n + block.x - 1) / block.x);
    
    // 전역 SwiGLU 융합 커널 실행
    swiglu_fusion_kernel<ElementC_, ElementD_><<<grid, block, 0, stream>>>(
        linear_ptr,
        gate_ptr,
        output_ptr,
        bias_linear_ptr,
        bias_gate_ptr,
        m,
        n,
        static_cast<float>(output_scale));
        
    return cudaGetLastError() == cudaSuccess ? 
        cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

}  // namespace cutlass_kernels
}  // namespace kernels  
}  // namespace tensorrt_llm