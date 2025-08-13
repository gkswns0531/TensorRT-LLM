# GPT-OSS on TensorRT-LLM

## Overview

GPT-OSS is a reasoning model with Mixture of Experts (MoE) architecture. The MoE weights are pre-quantized with MXFP4, while other weights use standard precision formats.

## Convert Checkpoints

### Standard precision (BF16)
```bash
python examples/models/core/gpt_oss/convert_checkpoint.py \
  --model_dir /path/to/gpt-oss-20b \
  --output_dir /path/to/tllm_ckpt \
  --dtype bfloat16
```

### MXFP4 + FP8 quantization (Recommended for GPT-OSS MoE)
```bash
python examples/models/core/gpt_oss/convert_checkpoint.py \
  --model_dir /path/to/gpt-oss-20b \
  --output_dir /path/to/tllm_fp8_ckpt \
  --dtype bfloat16 \
  --use_mxfp4_fp8 \
  --fp8_kv_cache
```

### MXFP4 + FP8 quantization with KV cache
```bash
python examples/models/core/gpt_oss/convert_checkpoint.py \
  --model_dir /path/to/gpt-oss-20b \
  --output_dir /path/to/tllm_fp8_ckpt \
  --dtype bfloat16 \
  --use_mxfp4_fp8 \
  --fp8_kv_cache
```

## MXFP4 + FP8 Quantization Features

- **MoE native support**: Direct MXFP4 handling without conversion overhead
- **Hybrid quantization**: MXFP4 for MoE experts, FP8 for other layers
- **No calibration required**: Weight-only quantization without dataset dependency
- **High accuracy**: Optimized quantization scheme for reasoning models
- **Fast conversion**: Single-command checkpoint conversion
- **Full TP support**: Tensor parallelism compatible
- **Hardware optimized**: Designed for H100/L4 MoE acceleration

## Build TensorRT Engine

```bash
trtllm-build \
  --checkpoint_dir /path/to/tllm_ckpt \
  --output_dir /path/to/engine \
  --max_batch_size 4 \
  --max_input_len 4096 \
  --max_seq_len 4096 \
  --kv_cache_type paged
```

## Serve Model

```bash
trtllm-serve /path/to/engine --backend tensorrt --tp_size 1 --ep_size 1
```

## MoE Support Matrix

The MoE weights are pre-quantized to MXFP4. Activations can be in BF16 (Hopper) or MXFP8 (Blackwell).

| Device | Activation | Weight | Supported moe_backend |
|---------|-----------|---------|-------------------|
| Hopper | BF16 | MXFP4 | **TRITON**, CUTLASS |
| Blackwell | MXFP8 | MXFP4 | CUTLASS, TRTLLM |

| moe_backend | TP | EP | AlltoAll |
|-------------|-----|-----|----------|
| CUTLASS | Yes | Yes | Yes |
| TRTLLM | Yes | Yes | No |
| TRITON | No | Yes | No |

**Performance Recommendations:**
- **Hopper**: Use `TRITON` for both latency and throughput
- **Blackwell**: Use `CUTLASS` for throughput, `TRTLLM` for latency

## Function Calling Support

GPT-OSS supports OpenAI-compatible function calling with XGrammar structural generation.

### Setup Server with XGrammar
```bash
cat > ./extra_llm_api_options.yaml <<EOF
guided_decoding_backend: xgrammar
EOF

trtllm-serve <model> \
    --backend pytorch \
    --extra_llm_api_options extra_llm_api_options.yaml
```

### Run Function Calling Example
```bash
python openai_chat_client_function_calling.py \
    --model <model> \
    --prompt "What is the weather like in SF?"
```

The function calling process:
1. **Function Selection**: LLM selects appropriate function and generates arguments
2. **Function Execution**: Client executes function with generated arguments  
3. **Response Generation**: LLM provides final response based on function results

## Technical Notes

- **MoE Architecture**: 8 experts per layer with top-2 routing
- **MXFP4 Weights**: Automatic detection and conversion to FP8 during native quantization
- **Attention Sinks**: Optional feature, can be disabled if needed
- **Long Context**: Adjust `max_seq_len` and KV cache settings as needed
- **Memory Efficiency**: FP8 quantization reduces memory usage by ~50%

## Troubleshooting

### Common Issues

**GPU Memory**: Use FP8 quantization to reduce memory requirements
```bash
# Add FP8 flags for memory efficiency
--use_fp8_rowwise --fp8_kv_cache
```

**Tensor Parallelism**: Ensure TP size matches available GPUs
```bash
# Example for 2 GPUs
--tp_size 2
```

**Long Sequences**: Increase memory allocation and adjust cache settings
```bash
--max_seq_len 8192 --kv_cache_type paged
```