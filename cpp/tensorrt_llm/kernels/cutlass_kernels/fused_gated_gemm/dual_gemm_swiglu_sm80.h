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
    
    // Linear GEMM 설정: A @ B_linear
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
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
        Stages>;
    
    // Gate GEMM 설정: A @ B_gate  
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
        cutlass::epilogue::thread::LinearCombination<
            ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator, ElementCompute>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages>;

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
        cutlass::TensorRef<ElementC const, LayoutC> ref_C;
        cutlass::TensorRef<ElementD, LayoutD> ref_D;
        typename LinearGemm::EpilogueOutputOp::Params linear_epilogue;
        typename GateGemm::EpilogueOutputOp::Params gate_epilogue;
        
        // 생성자
        Arguments(cutlass::gemm::GemmCoord const& problem_size,
                  cutlass::TensorRef<ElementA const, LayoutA> ref_A,
                  cutlass::TensorRef<ElementB const, LayoutB> ref_B,  // 전체 B
                  cutlass::TensorRef<ElementC const, LayoutC> ref_C,
                  cutlass::TensorRef<ElementD, LayoutD> ref_D,
                  ElementCompute alpha = ElementCompute(1),
                  ElementCompute beta = ElementCompute(0))
            : problem_size(problem_size)
            , ref_A(ref_A)
            , ref_C(ref_C)
            , ref_D(ref_D)
            , linear_epilogue({alpha, beta})
            , gate_epilogue({alpha, ElementCompute(0)})  // gate는 bias 없이
        {
            // B 매트릭스 분할: [B_linear | B_gate]
            int m = problem_size.m();
            int n = problem_size.n();  // 이미 n/2
            int k = problem_size.k();
            
            // B_linear: 첫 번째 n/2 columns (0 ~ n-1)
            ref_B_linear = cutlass::TensorRef<ElementB const, LayoutB>(
                ref_B.data(), 
                cutlass::layout::RowMajor::packed({k, n}));
            
            // B_gate: 두 번째 n/2 columns (n ~ 2n-1)  
            ref_B_gate = cutlass::TensorRef<ElementB const, LayoutB>(
                ref_B.data() + k * n,  // n 컬럼만큼 오프셋
                cutlass::layout::RowMajor::packed({k, n}));
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
            args.ref_C,
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

        // 3. SwiGLU 융합: linear * SiLU(gate) → final_output
        return launch_swiglu_fusion_impl(
            linear_output, gate_output, args.ref_D.data(), m, n, stream);
    }

private:

    // SwiGLU 융합 커널 런처 (전역 함수 호출)
    template<typename ElementC_, typename ElementD_>
    cutlass::Status launch_swiglu_fusion_impl(
        ElementC_ const* linear_ptr,
        ElementC_ const* gate_ptr, 
        ElementD_* output_ptr,
        int m, int n,
        cudaStream_t stream);
};

// SwiGLU 융합 커널 - 클래스 외부 전역 함수로 정의
template<typename ElementC, typename ElementD>
__global__ void swiglu_fusion_kernel(
    ElementC const* __restrict__ linear_ptr,
    ElementC const* __restrict__ gate_ptr, 
    ElementD* __restrict__ output_ptr,
    int m, int n) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = m * n;
    
    if (tid < total_elements) {
        ElementC linear_val = linear_ptr[tid];
        ElementC gate_val = gate_ptr[tid];
        
        // SwiGLU: linear * SiLU(gate)
        float gate_f = static_cast<float>(gate_val);
        float sigmoid_gate = gate_f / (1.0f + expf(-gate_f));  // SiLU
        float linear_f = static_cast<float>(linear_val);
        
        output_ptr[tid] = static_cast<ElementD>(linear_f * sigmoid_gate);
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
        int m, int n,
        cudaStream_t stream) {
    
    // GPU 커널 설정
    dim3 block(256);
    dim3 grid((m * n + block.x - 1) / block.x);
    
    // 전역 SwiGLU 융합 커널 실행
    swiglu_fusion_kernel<ElementC_, ElementD_><<<grid, block, 0, stream>>>(
        linear_ptr, gate_ptr, output_ptr, m, n);
        
    return cudaGetLastError() == cudaSuccess ? 
        cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

}  // namespace cutlass_kernels
}  // namespace kernels  
}  // namespace tensorrt_llm