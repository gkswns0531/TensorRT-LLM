"""
Weights conversion utilities for GPT-OSS to TensorRT-LLM checkpoint format.

Features:
- Load HF checkpoints (safetensors)
- Extract/reshape Q/K/V/O projections with GQA and TP
- Prepare MoE (gate_up_proj/down_proj) as either BF16 (dequantized) or MXFP4 (kept as FP4 blocks with NVFP4 aux scales)
- Extract attention sinks per layer and slice by TP

This converter writes a single shard per call (rank-aware) and avoids custom op dependencies at convert-time.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import safetensors
from safetensors import safe_open
import torch.nn.functional as F
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.convert_utils import split_matrix_tp, dup_kv_weight, dup_kv_bias
from tensorrt_llm._utils import str_dtype_to_torch
from .config import GptOssConfig
from tensorrt_llm.quantization.mode import QuantAlgo


_STREAM_TILE_ROWS = 1024


@dataclass
class ConvertContext:
    config: GptOssConfig
    mapping: Mapping
    quant_config: QuantConfig
    model_dir: Path
    use_hf: bool


def _detect_hf_or_original(model_dir: Union[str, Path]) -> Tuple[bool, Path]:
    p = Path(model_dir)
    index = p / 'model.safetensors.index.json'
    if not index.exists():
        raise FileNotFoundError(
            f"Hugging Face index file not found: {index}. GPT-OSS converter only supports HF layout."
        )
    return (True, p)


def load_hf_model(model_dir: Path):
    import transformers
    logger.info("Loading HF model for GPT-OSS")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype='auto')
    model.eval()
    return model


def convert_and_save(
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: GptOssConfig,
    *,
    quant_config: Optional[QuantConfig] = None,
    moe_export: str = 'auto',  # {'auto','mxfp4','bf16','fp16'}
    target_arch: Optional[str] = None,
    nvfp4_scale_mode: str = 'heuristic',  # {'heuristic','ones','auto'}
    stream_tile_rows: int = _STREAM_TILE_ROWS,
    interleave_scales: bool = False,
) -> None:
    """
    Convert the weights into TensorRT-LLM checkpoint shards (rank*.safetensors).

    This initial version writes config only, and stubs empty rank0 weight files
    to bootstrap the pipeline. Weight population will be added incrementally.
    """
    use_hf, p_model = _detect_hf_or_original(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Get target dtype from config (same as Llama pattern)
    torch_dtype = str_dtype_to_torch(config.dtype)
    logger.info(f"Using target dtype: {torch_dtype}")

    # Decide MoE export mode
    moe_mode = moe_export
    if moe_export == 'auto':
        arch = (target_arch or '').lower()
        if arch in ('sm80', 'sm_80', 'a100', 'sm89', 'sm_89', 'l4'):
            moe_mode = 'fp16'
        else:
            moe_mode = 'mxfp4'
    elif moe_export in ('mxfp4', 'bf16', 'fp16'):
        moe_mode = moe_export

    # Force RoPE type that TRT plugin recognizes
    config.position_embedding_type = "rope_gpt_neox"

    # Attach quantization hint into config: NVFP4 when exporting MXFP4
    if moe_mode == 'mxfp4':
        # Preserve caller-provided quant_config if any; otherwise set NVFP4
        quant_cfg = quant_config if quant_config is not None else QuantConfig(quant_algo=QuantAlgo.NVFP4)
        config.quantization = quant_cfg
    # Save initial config.json (may be overwritten after dimension inference)
    config.to_json_file(str(output / 'config.json'))

    # Prepare minimal shards and try to import attention sinks
    world_size = config.mapping.world_size if config.mapping else 1
    sinks_per_layer = {}
    sinks_per_layer = _extract_sinks_from_index(p_model)
    logger.info(f"Loaded sinks for {len(sinks_per_layer)} layers")

    # Load weight map (HF: from index; original: from single file)
    weight_map = None
    original_single_file = None
    cache_files: Dict[str, Dict[str, torch.Tensor]] = {}
    # HF only
    with open(p_model / 'model.safetensors.index.json', 'r') as f:
        idx = json.load(f)
    weight_map = idx['weight_map']

    def load_file(file_name: str) -> Dict[str, torch.Tensor]:
        if file_name in cache_files:
            return cache_files[file_name]
        data_path = p_model / file_name
        # Use safe_open instead of safetensors.torch.load_file for better compatibility
        tensors = {}
        with safe_open(data_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        cache_files[file_name] = tensors
        return tensors

    tp_size = config.mapping.tp_size if config.mapping else 1

    def _nvfp4_aux_from_mxfp4_scales(scales: torch.Tensor,
                                     out_features: int,
                                     in_features: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate NVFP4 auxiliary scale tensors from MXFP4 scales.

        Returns (weights_block_scaling_factor, weights_block_scaling_factor_interleaved, activation_global_scaling_factor, alpha)
        Shapes:
          - wbsf, wbsf_interleaved: [E, out_features_pad, in_features_pad/16]
          - act_gsf: [1]
          - alpha: [E]
        """
        E = scales.shape[0]
        vec = 16
        # pad columns to ceil(in_features/vec)
        want_cols = math.ceil(in_features / vec)
        have_cols = scales.shape[-1]
        if have_cols < want_cols:
            scales = F.pad(scales, (0, want_cols - have_cols))
        # pad rows to multiple of 128 as NVFP4 plugin prefers
        want_rows = math.ceil(out_features / 128) * 128
        have_rows = scales.shape[-2]
        if have_rows < want_rows:
            pad_rows = want_rows - have_rows
            pad_tensor = torch.zeros((E, pad_rows, scales.shape[-1]), dtype=scales.dtype)
            scales = torch.cat([scales, pad_tensor], dim=-2)

        # Select scale mode
        mode = nvfp4_scale_mode or 'heuristic'
        if mode == 'auto':
            mode = 'heuristic'

        if mode == 'ones':
            sf_fp8 = torch.ones_like(scales, dtype=torch.float8_e4m3fn)
        else:
            # cast to fp8 for weights_block_scaling_factor
            sf_fp8 = scales.to(torch.float8_e4m3fn)

        # Interleaved layout key: either defer to loader or interleave now
        if interleave_scales:
            inter_u8 = torch.ops.trtllm.block_scale_interleave(sf_fp8.view(torch.uint8).contiguous())
            inter = inter_u8.view(sf_fp8.dtype)
        else:
            # Defer interleave to loader-side
            inter = sf_fp8.clone()

        # Global activation scale and alpha
        act_gsf = torch.ones((1, ), dtype=torch.float32)
        a = torch.ones((E, ), dtype=torch.float32)
        return sf_fp8, inter, act_gsf, a

    def _dequantize_mxfp4(blocks: torch.Tensor,
                           scales: torch.Tensor,
                           *,
                           dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        MXFP4 blocks+scales to FP 

        Supported shapes:
          - blocks: [E, OUT, IN/vec] (uint8)
          - blocks: [E, OUT, IN/vec, PACK] (uint8)  # PACK size example: 16
          - scales: [E, OUT, IN/vec] (FP32/FP16/BF16/UINT8)

        Process overview:
          1) Flatten blocks to byte units ([E, OUT, IN/vec * PACK_BYTES])
          2) Unpack each byte to 4bit pairs (low/high) → length 2x
          3) Broadcast scales to byte·nibble extended length
          4) element-wise multiply, cast to target dtype
        Result shape: [E, OUT, IN] (IN = IN/vec * PACK_BYTES * 2)
        """
        if blocks.dim() < 3 or scales.dim() < 3:
            raise ValueError(
                f"blocks/scales must be at least 3D, got blocks{tuple(blocks.shape)} scales{tuple(scales.shape)}"
            )

        if blocks.dim() == 4:
            E, out_rows, in_cols, pack_bytes = blocks.shape
            blocks_bytes = blocks.reshape(E, out_rows, in_cols * pack_bytes)
            byte_repeat = pack_bytes
        elif blocks.dim() == 3:
            E, out_rows, in_cols = blocks.shape
            blocks_bytes = blocks
            byte_repeat = 1
        else:
            leading = int(torch.tensor(blocks.shape[:-2]).prod().item())
            blocks = blocks.reshape(leading, blocks.shape[-2], blocks.shape[-1])
            E, out_rows, in_cols = blocks.shape
            blocks_bytes = blocks
            byte_repeat = 1

        if scales.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            sf = scales.to(torch.float32) / 255.0
        else:
            sf = scales.to(torch.float32)

        blk = blocks_bytes.view(torch.uint8)
        low = (blk & 0x0F).to(torch.int8)
        high = ((blk >> 4) & 0x0F).to(torch.int8)
        low = (low ^ 0x08) - 0x08
        high = (high ^ 0x08) - 0x08
        
        depacked = torch.stack([low, high], dim=-1).view(E, out_rows, -1)

        sf = sf.unsqueeze(-1).repeat(1, 1, 1, byte_repeat * 2).view(
            E, out_rows, -1)

        deq = (depacked.to(torch.float32) * sf).to(dtype)
        return deq

    def _dequantize_mxfp4_streaming(blocks: torch.Tensor,
                                     scales: torch.Tensor,
                                     *,
                                     dtype: torch.dtype,
                                      tile_rows: int = _STREAM_TILE_ROWS) -> torch.Tensor:
        # shapes: blocks [E, OUT, IN/vec] or [E, OUT, IN/vec, PACK], scales [E, OUT, IN/vec]
        assert blocks.dim() in (3, 4), f"blocks must be 3D/4D, got {tuple(blocks.shape)}"
        assert scales.dim() == 3, f"scales must be 3D, got {tuple(scales.shape)}"

        if blocks.dim() == 4:
            E, out_rows, in_cols, pack_bytes = blocks.shape
            byte_repeat = pack_bytes
            in_features = in_cols * pack_bytes * 2
            assert (E, out_rows, in_cols) == (scales.shape[0], scales.shape[1], scales.shape[2])
        else:
            E, out_rows, in_cols = blocks.shape
            byte_repeat = 1
            in_features = in_cols * 2
            assert blocks.shape == scales.shape, f"blocks/scales shape mismatch: {tuple(blocks.shape)} vs {tuple(scales.shape)}"

        out = torch.empty((E, out_rows, in_features), dtype=dtype)

        for r0 in range(0, out_rows, tile_rows):
            r1 = min(out_rows, r0 + tile_rows)
            if blocks.dim() == 4:
                b_bytes = blocks[:, r0:r1, :, :].reshape(E, r1 - r0, in_cols * byte_repeat)
            else:
                b_bytes = blocks[:, r0:r1, :]

            if scales.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                sf = scales[:, r0:r1, :].to(torch.float32) / 255.0
            else:
                sf = scales[:, r0:r1, :].to(torch.float32)

            blk = b_bytes.view(torch.uint8)
            low = (blk & 0x0F).to(torch.int8)
            high = ((blk >> 4) & 0x0F).to(torch.int8)
            low = (low ^ 0x08) - 0x08
            high = (high ^ 0x08) - 0x08
            depacked = torch.stack([low, high], dim=-1).view(E, r1 - r0, -1).to(torch.float32)

            sf = sf.unsqueeze(-1).repeat(1, 1, 1, byte_repeat * 2).view(E, r1 - r0, -1)
            out[:, r0:r1, :] = (depacked * sf).to(dtype)

        return out

    # Generate a single shard for current mapping.rank
    rank = config.mapping.rank if config.mapping else 0
    shard_path = output / f'rank{rank}.safetensors'
    weights: Dict[str, torch.Tensor] = {}

    # Distribute sinks across TP ranks
    if sinks_per_layer:
        tp_size = config.mapping.tp_size if config.mapping else 1
        num_heads = config.num_attention_heads
        heads_per_rank = num_heads // tp_size if tp_size > 0 else num_heads

        for layer_idx, sinks in sinks_per_layer.items():
            # sinks shape: [num_heads]
            beg = rank * heads_per_rank
            end = beg + heads_per_rank
            tp_sinks = sinks[beg:end].contiguous().to(torch.float32)
            key = f"transformer.layers.{layer_idx}.attention.sinks"
            weights[key] = tp_sinks

    # Extract attention/projection/ln/emb/lm_head when index exists
    if weight_map is not None:
        nl = config.num_hidden_layers
        # Per-layer
        # Infer and correct dimension config from weights of the first layer
        #  - hidden_size/head_size from q_proj.rows
        #  - intermediate_size from gate_up_proj_bias
        i0 = 0
        def get(name: str) -> torch.Tensor:
            file = weight_map.get(name)
            if file is None:
                raise KeyError(name)
            tensors = load_file(file)
            return tensors[name]

        q0 = get(f'model.layers.{i0}.self_attn.q_proj.weight')
        k0 = get(f'model.layers.{i0}.self_attn.k_proj.weight')
        q_rows = q0.shape[0]
        k_rows = k0.shape[0]
        nh = config.num_attention_heads
        if q_rows % nh != 0:
            raise ValueError(f"q_proj.rows {q_rows} not divisible by num_attention_heads {nh}")
        implied_head = q_rows // nh
        # hidden_size remains as declared in config.json (e.g., 2880). q_rows reflects heads*head_dim (e.g., 4096).
        if config.head_size != implied_head:
            logger.info(f"Adjusting head_size from {config.head_size} to {implied_head} based on q_proj.rows/num_heads")
            config.head_size = implied_head

        # Infer num_key_value_heads from k_proj.rows and head_size
        if k_rows % config.head_size != 0:
            raise ValueError(f"k_proj.rows {k_rows} not divisible by head_size {config.head_size}")
        implied_kv_heads = k_rows // config.head_size
        if config.num_key_value_heads != implied_kv_heads:
            logger.info(f"Adjusting num_key_value_heads from {config.num_key_value_heads} to {implied_kv_heads} based on k_proj.rows/head_size")
            config.num_key_value_heads = implied_kv_heads

        # Infer intermediate_size from gate_up_proj_bias (expected = 2 * intermediate_size)
        gub = get(f'model.layers.{i0}.mlp.experts.gate_up_proj_bias')
        if gub.shape[-1] % 2 != 0:
            raise ValueError(f"gate_up_proj_bias length {gub.shape[-1]} not even; cannot infer intermediate_size")
        implied_inter = gub.shape[-1] // 2
        if config.intermediate_size != implied_inter:
            logger.info(f"Adjusting intermediate_size from {config.intermediate_size} to {implied_inter} based on gate_up_proj_bias")
            config.intermediate_size = implied_inter

        # Persist updated dimension config
        config.to_json_file(str(output / 'config.json'))

        # Recompute implied head again after adjustments
        tp_size = config.mapping.tp_size if config.mapping else 1
        nl = config.num_hidden_layers
        # Set dequantization dtype for non-MXFP4 export
        deq_dtype = None
        if moe_mode == 'bf16':
            deq_dtype = torch.bfloat16
        elif moe_mode == 'fp16':
            deq_dtype = torch.float16

        # Per-layer
        for i in range(nl):
                # q,k,v
                def get(name: str) -> torch.Tensor:
                    file = weight_map.get(name)
                    if file is None:
                        raise KeyError(name)
                    tensors = load_file(file)
                    return tensors[name]

                q_w = get(f'model.layers.{i}.self_attn.q_proj.weight').to(torch_dtype)
                k_w = get(f'model.layers.{i}.self_attn.k_proj.weight').to(torch_dtype)
                v_w = get(f'model.layers.{i}.self_attn.v_proj.weight').to(torch_dtype)
                q_b = get(f'model.layers.{i}.self_attn.q_proj.bias').to(torch_dtype)
                k_b = get(f'model.layers.{i}.self_attn.k_proj.bias').to(torch_dtype)
                v_b = get(f'model.layers.{i}.self_attn.v_proj.bias').to(torch_dtype)

                # shape checks for QKV
                nh = config.num_attention_heads
                nkh = config.num_key_value_heads
                hs = config.head_size
                assert q_w.shape[0] == nh * hs, f"q_proj.rows {q_w.shape[0]} != num_heads*head_size {nh*hs}"
                assert k_w.shape[0] == nkh * hs, f"k_proj.rows {k_w.shape[0]} != num_kv_heads*head_size {nkh*hs}"
                assert v_w.shape[0] == nkh * hs, f"v_proj.rows {v_w.shape[0]} != num_kv_heads*head_size {nkh*hs}"

                if tp_size == 1:
                    qkv_w_all = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
                    qkv_b_all = torch.cat([q_b, k_b, v_b], dim=0).contiguous()
                    weights[f'transformer.layers.{i}.attention.qkv.weight'] = qkv_w_all
                    weights[f'transformer.layers.{i}.attention.qkv.bias'] = qkv_b_all
                else:
                    # GQA-aware TP split: split Q per num_heads, duplicate KV per num_kv_heads
                    num_heads = config.num_attention_heads
                    num_kv_heads = config.num_key_value_heads
                    head_size = config.head_size

                    # Q rows split evenly across TP ranks
                    q_w_tp = split_matrix_tp(q_w, tp_size, rank, dim=0)
                    q_b_tp = split_matrix_tp(q_b, tp_size, rank, dim=0)

                    # Duplicate KV rows if needed so rows % tp_size == 0
                    k_w_eff = k_w
                    v_w_eff = v_w
                    k_b_eff = k_b
                    v_b_eff = v_b
                    if (k_w.shape[0] % tp_size) != 0 and num_kv_heads > 0:
                        k_w_eff = dup_kv_weight(k_w, num_kv_heads, tp_size)
                        v_w_eff = dup_kv_weight(v_w, num_kv_heads, tp_size)
                        k_b_eff = dup_kv_bias(k_b, num_kv_heads, tp_size)
                        v_b_eff = dup_kv_bias(v_b, num_kv_heads, tp_size)

                    k_w_tp = split_matrix_tp(k_w_eff, tp_size, rank, dim=0)
                    v_w_tp = split_matrix_tp(v_w_eff, tp_size, rank, dim=0)
                    k_b_tp = split_matrix_tp(k_b_eff, tp_size, rank, dim=0)
                    v_b_tp = split_matrix_tp(v_b_eff, tp_size, rank, dim=0)

                    qkv_w_tp = torch.cat([q_w_tp, k_w_tp, v_w_tp], dim=0).contiguous()
                    qkv_b_tp = torch.cat([q_b_tp, k_b_tp, v_b_tp], dim=0).contiguous()

                    weights[f'transformer.layers.{i}.attention.qkv.weight'] = qkv_w_tp
                    weights[f'transformer.layers.{i}.attention.qkv.bias'] = qkv_b_tp

                o_w = get(f'model.layers.{i}.self_attn.o_proj.weight').to(torch_dtype)
                o_b = get(f'model.layers.{i}.self_attn.o_proj.bias').to(torch_dtype)
                assert o_w.shape[1] == nh * hs, f"o_proj.cols {o_w.shape[1]} != num_heads*head_size {nh*hs}"
                if tp_size == 1:
                    weights[f'transformer.layers.{i}.attention.dense.weight'] = o_w.contiguous()
                    weights[f'transformer.layers.{i}.attention.dense.bias'] = o_b.contiguous()
                else:
                    # dense is row-parallel in many models; split columns for output gathering
                    tp_w = torch.chunk(o_w, tp_size, dim=1)[rank].contiguous()
                    weights[f'transformer.layers.{i}.attention.dense.weight'] = tp_w
                    # Write bias on all ranks; non-zero ranks will be zeroed in preprocess
                    weights[f'transformer.layers.{i}.attention.dense.bias'] = o_b.contiguous()

                in_ln = get(f'model.layers.{i}.input_layernorm.weight').to(torch_dtype)
                po_ln = get(f'model.layers.{i}.post_attention_layernorm.weight').to(torch_dtype)
                weights[f'transformer.layers.{i}.input_layernorm.weight'] = in_ln.contiguous()
                weights[f'transformer.layers.{i}.post_layernorm.weight'] = po_ln.contiguous()

                # MoE (raw MXFP4) passthrough for later consumption
                gate_up_blocks = get(f'model.layers.{i}.mlp.experts.gate_up_proj_blocks')
                gate_up_scales = get(f'model.layers.{i}.mlp.experts.gate_up_proj_scales')
                gate_up_bias = get(f'model.layers.{i}.mlp.experts.gate_up_proj_bias')
                # shape checks for gate_up blocks/scales
                if gate_up_blocks.dim() == 4:
                    assert gate_up_scales.dim() == 3
                    assert gate_up_blocks.shape[0] == gate_up_scales.shape[0]
                    assert gate_up_blocks.shape[1] == gate_up_scales.shape[1]
                    assert gate_up_blocks.shape[2] == gate_up_scales.shape[2]
                elif gate_up_blocks.dim() == 3:
                    assert gate_up_scales.dim() == 3
                    assert gate_up_blocks.shape == gate_up_scales.shape
                else:
                    assert False, f"Unexpected gate_up_blocks dim: {gate_up_blocks.dim()}"

                # Deinterleave gate/up along the OUT feature axis for blocks/scales.
                # blocks may be 4D: [E, OUT, IN/vec, pack_vec], scales 3D: [E, OUT, IN/vec]
                def _deinterleave_gate_up(tensor, out_axis):
                    out_len = tensor.shape[out_axis]
                    idx_even = torch.arange(0, out_len, 2)
                    idx_odd = torch.arange(1, out_len, 2)
                    take = lambda t, idx: t.index_select(out_axis, idx.to(t.device))
                    return take(tensor, idx_even), take(tensor, idx_odd)

                # Determine out axis for blocks/scales
                out_axis_blocks = -3 if gate_up_blocks.dim() >= 4 else -2
                out_axis_scales = -2  # scales expected [E, OUT, IN/vec]

                gate_w, up_w = _deinterleave_gate_up(gate_up_blocks,
                                                     out_axis_blocks)
                gate_sc, up_sc = _deinterleave_gate_up(gate_up_scales,
                                                       out_axis_scales)

                fc_blocks = torch.cat([gate_w, up_w], dim=out_axis_blocks).contiguous()
                fc_scales = torch.cat([gate_sc, up_sc], dim=out_axis_scales).contiguous()
                # Bias: interleaved pairs gate/up across last dim
                gate_b = gate_up_bias[:, ::2]
                up_b = gate_up_bias[:, 1::2]
                fc_bias = torch.cat([gate_b, up_b], dim=-1).contiguous()
                # Cast bias to target dtype depending on moe_mode
                if moe_mode == 'mxfp4':
                    fc_bias = fc_bias.to(torch_dtype)
                else:
                    fc_bias = fc_bias.to(deq_dtype)

                def _collapse_pack_dim(blocks: torch.Tensor) -> torch.Tensor:
                    if blocks.dim() == 4:
                        E, OUT, IN_DIV, PACK = blocks.shape
                        return blocks.reshape(E, OUT, IN_DIV * PACK)
                    return blocks

                if moe_mode == 'mxfp4':
                    if tp_size == 1:
                        fc_blocks_ckpt = _collapse_pack_dim(fc_blocks)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias

                        fc_out_features = fc_blocks.shape[1]
                        fc_in_features = config.hidden_size
                        wbsf, wbsf_inter, act_gsf, alpha = _nvfp4_aux_from_mxfp4_scales(
                            fc_scales, out_features=fc_out_features, in_features=fc_in_features)
                        weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor'] = wbsf
                        weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor_interleaved'] = wbsf_inter
                        weights[f'transformer.layers.{i}.mlp.fc.activation_global_scaling_factor'] = act_gsf
                        weights[f'transformer.layers.{i}.mlp.fc.alpha'] = alpha
                    else:
                        out_axis_blocks_tp = -3 if fc_blocks.dim() == 4 else -2
                        fc_blocks_tp = torch.chunk(fc_blocks, tp_size, dim=out_axis_blocks_tp)[rank].contiguous()
                        fc_scales_tp = torch.chunk(fc_scales, tp_size, dim=-2)[rank].contiguous()
                        fc_bias_tp = torch.chunk(fc_bias, tp_size, dim=-1)[rank].contiguous()

                        fc_blocks_ckpt = _collapse_pack_dim(fc_blocks_tp)
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias_tp

                        fc_out_features_tp = fc_blocks_tp.shape[1]
                        fc_in_features = config.hidden_size
                        wbsf, wbsf_inter, act_gsf, alpha = _nvfp4_aux_from_mxfp4_scales(
                            fc_scales_tp, out_features=fc_out_features_tp, in_features=fc_in_features)
                        weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor'] = wbsf
                        weights[f'transformer.layers.{i}.mlp.fc.weights_block_scaling_factor_interleaved'] = wbsf_inter
                        weights[f'transformer.layers.{i}.mlp.fc.activation_global_scaling_factor'] = act_gsf
                        weights[f'transformer.layers.{i}.mlp.fc.alpha'] = alpha
                else:
                    # Offline dequantization to desired float dtype (bf16/fp16)
                    assert deq_dtype is not None, f"Unsupported moe_mode {moe_mode}"
                    if tp_size == 1:
                        fc_weight = _dequantize_mxfp4_streaming(fc_blocks, fc_scales, dtype=deq_dtype, tile_rows=stream_tile_rows)
                        expected_in_fc = config.hidden_size
                        if fc_weight.shape[-1] > expected_in_fc:
                            fc_weight = fc_weight[..., :expected_in_fc]
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_weight.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias
                    else:
                        out_axis_blocks_tp = -3 if fc_blocks.dim() == 4 else -2
                        fc_blocks_tp = torch.chunk(fc_blocks, tp_size, dim=out_axis_blocks_tp)[rank].contiguous()
                        fc_scales_tp = torch.chunk(fc_scales, tp_size, dim=-2)[rank].contiguous()
                        fc_bias_tp = torch.chunk(fc_bias, tp_size, dim=-1)[rank].contiguous()
                        fc_weight_tp = _dequantize_mxfp4_streaming(fc_blocks_tp, fc_scales_tp, dtype=deq_dtype, tile_rows=stream_tile_rows)
                        expected_in_fc_tp = config.hidden_size
                        if fc_weight_tp.shape[-1] > expected_in_fc_tp:
                            fc_weight_tp = fc_weight_tp[..., :expected_in_fc_tp]
                        weights[f'transformer.layers.{i}.mlp.fc.weight'] = fc_weight_tp.contiguous()
                        weights[f'transformer.layers.{i}.mlp.fc.bias'] = fc_bias_tp

                down_blocks = get(f'model.layers.{i}.mlp.experts.down_proj_blocks')
                down_scales = get(f'model.layers.{i}.mlp.experts.down_proj_scales')
                down_bias = get(f'model.layers.{i}.mlp.experts.down_proj_bias')
                if down_blocks.dim() == 4:
                    assert down_scales.dim() == 3
                    assert down_blocks.shape[0] == down_scales.shape[0]
                    assert down_blocks.shape[1] == down_scales.shape[1]
                    assert down_blocks.shape[2] == down_scales.shape[2]
                elif down_blocks.dim() == 3:
                    assert down_scales.dim() == 3
                    assert down_blocks.shape == down_scales.shape
                else:
                    assert False, f"Unexpected down_blocks dim: {down_blocks.dim()}"
                # Keep 3D layout [E, out, in/vec] for dequantization
                # Align down_proj shapes: allow 4D blocks [E, OUT, IN/vec, pack_vec] and 3D scales [E, OUT, IN/vec]
                proj_blocks = down_blocks.contiguous()
                proj_scales = down_scales.contiguous()
                if proj_blocks.dim() == 4 and proj_scales.dim() == 3:
                    # no change needed for dequant path; helper handles vec packing
                    pass

                if moe_mode == 'mxfp4':
                    if tp_size == 1:
                        proj_blocks_ckpt = _collapse_pack_dim(proj_blocks)
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(torch_dtype).contiguous()

                        proj_out_features = proj_blocks.shape[1]
                        proj_in_features = config.intermediate_size
                        wbsf, wbsf_inter, act_gsf, alpha = _nvfp4_aux_from_mxfp4_scales(
                            proj_scales, out_features=proj_out_features, in_features=proj_in_features)
                        weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor'] = wbsf
                        weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor_interleaved'] = wbsf_inter
                        weights[f'transformer.layers.{i}.mlp.proj.activation_global_scaling_factor'] = act_gsf
                        weights[f'transformer.layers.{i}.mlp.proj.alpha'] = alpha
                    else:
                        in_axis_blocks_tp = -2 if proj_blocks.dim() == 4 else -1
                        proj_blocks_tp = torch.chunk(proj_blocks, tp_size, dim=in_axis_blocks_tp)[rank].contiguous()
                        proj_scales_tp = torch.chunk(proj_scales, tp_size, dim=-1)[rank].contiguous()

                        proj_blocks_ckpt = _collapse_pack_dim(proj_blocks_tp)
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_blocks_ckpt.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(torch_dtype).contiguous()

                        proj_out_features = proj_blocks.shape[1]
                        proj_in_features_tp = max(1, config.intermediate_size // tp_size)
                        wbsf, wbsf_inter, act_gsf, alpha = _nvfp4_aux_from_mxfp4_scales(
                            proj_scales_tp, out_features=proj_out_features, in_features=proj_in_features_tp)
                        weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor'] = wbsf
                        weights[f'transformer.layers.{i}.mlp.proj.weights_block_scaling_factor_interleaved'] = wbsf_inter
                        weights[f'transformer.layers.{i}.mlp.proj.activation_global_scaling_factor'] = act_gsf
                        weights[f'transformer.layers.{i}.mlp.proj.alpha'] = alpha
                else:
                    assert deq_dtype is not None, f"Unsupported moe_mode {moe_mode}"
                    if tp_size == 1:
                        proj_weight = _dequantize_mxfp4_streaming(proj_blocks, proj_scales, dtype=deq_dtype, tile_rows=stream_tile_rows)
                        expected_in = config.intermediate_size
                        if proj_weight.shape[-1] > expected_in:
                            proj_weight = proj_weight[..., :expected_in]
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_weight.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(deq_dtype).contiguous()
                    else:
                        in_axis_blocks_tp = -2 if proj_blocks.dim() == 4 else -1
                        proj_blocks_tp = torch.chunk(proj_blocks, tp_size, dim=in_axis_blocks_tp)[rank].contiguous()
                        proj_scales_tp = torch.chunk(proj_scales, tp_size, dim=-1)[rank].contiguous()
                        proj_weight_tp = _dequantize_mxfp4_streaming(proj_blocks_tp, proj_scales_tp, dtype=deq_dtype, tile_rows=stream_tile_rows)
                        expected_in_tp = max(1, config.intermediate_size // tp_size)
                        if proj_weight_tp.shape[-1] > expected_in_tp:
                            proj_weight_tp = proj_weight_tp[..., :expected_in_tp]
                        weights[f'transformer.layers.{i}.mlp.proj.weight'] = proj_weight_tp.contiguous()
                        weights[f'transformer.layers.{i}.mlp.proj.bias'] = down_bias.to(deq_dtype).contiguous()

                # BF16 per-expert path intentionally not supported for gpt-oss (MoE uses MXFP4)

        # Embeddings & final norm & lm_head (once per rank)
        emb_w = get('model.embed_tokens.weight').to(torch_dtype)
        if tp_size == 1:
            weights['transformer.vocab_embedding.weight'] = emb_w.contiguous()
        else:
            # column-sharded along vocab dimension
            tp_emb = torch.chunk(emb_w, tp_size, dim=0)[rank].contiguous()
            weights['transformer.vocab_embedding.weight'] = tp_emb

        ln_f = get('model.norm.weight').to(torch_dtype)
        weights['transformer.ln_f.weight'] = ln_f.contiguous()

        lm_w = get('lm_head.weight').to(torch_dtype)
        if tp_size == 1:
            weights['lm_head.weight'] = lm_w.contiguous()
        else:
            tp_lm = torch.chunk(lm_w, tp_size, dim=0)[rank].contiguous()
            weights['lm_head.weight'] = tp_lm

    # Router weight (optional; present in MoE) — write on all ranks
    if weight_map is not None:
        # probe existence
        _ = weight_map.get('model.layers.0.mlp.router.weight')
        for i in range(config.num_hidden_layers):
            name = f'model.layers.{i}.mlp.router.weight'
            file = weight_map.get(name)
            if file is None:
                continue
            tensors = load_file(file)
            if name not in tensors:
                continue
            w = tensors[name].to(torch_dtype)
            # keep unsplit; MOE handles distribution internally
            weights[f'transformer.layers.{i}.mlp.router.weight'] = w.contiguous()

    safetensors.torch.save_file(weights, str(shard_path))
    logger.info(f"Wrote shard with {len(weights)} tensors: {shard_path}")

    logger.info(
        f"convert_and_save: Successfully created shard rank{rank} (of world_size={world_size}) with QKV/GQA/TP, MoE MXFP4/BF16, and attention sinks support."
    )


# --- Implementation guides (to be filled in next iterations) ---

def _extract_qkv_from_hf(hf_model, layer_idx: int, config: GptOssConfig) -> Dict[str, torch.Tensor]:
    """TODO: Read HF layer q,k,v weights + bias; return raw tensors prior to TP split."""
    raise NotImplementedError


def _tp_split_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mapping: Mapping, num_heads: int,
                  num_kv_heads: int, head_size: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """TODO: Perform TP split and KV duplication for GQA. Return weight/bias dicts per TP rank."""
    raise NotImplementedError


def _extract_moe_blocks_and_scales(hf_weights: Dict[str, torch.Tensor], config: GptOssConfig) -> Dict[str, torch.Tensor]:
    """TODO: Handle MXFP4 blocks+scales for gate_up_proj/down_proj, deinterleave gate/up and prepare per-expert tensors."""
    raise NotImplementedError


def _load_attention_sinks(hf_model_or_dir: Union[str, Path], num_layers: int, num_heads: int,
                          mapping: Mapping) -> Dict[int, torch.Tensor]:
    """TODO: Load per-layer sinks (float32 per head) and TP-slice them."""
    raise NotImplementedError


def _extract_sinks_from_index(model_dir: Path) -> Dict[int, torch.Tensor]:
    """
    Parse HF safetensors index and extract per-layer self_attn.sinks tensors.
    Returns: dict[layer_idx] = 1D float tensor of length num_heads.
    """
    index_path = model_dir / 'model.safetensors.index.json'
    with open(index_path, 'r') as f:
        idx = json.load(f)  # type: ignore
    weight_map: Dict[str, str] = idx['weight_map']

    # group keys per layer
    sinks: Dict[int, torch.Tensor] = {}
    cache_files: Dict[str, Dict[str, torch.Tensor]] = {}

    def load_file(file_name: str) -> Dict[str, torch.Tensor]:
        if file_name in cache_files:
            return cache_files[file_name]
        data_path = model_dir / file_name
        # Use safe_open for better compatibility
        tensors = {}
        with safe_open(data_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        cache_files[file_name] = tensors
        return tensors

    for key, file_name in weight_map.items():
        # interested in keys like: model.layers.{i}.self_attn.sinks
        if not key.endswith('.self_attn.sinks'):
            continue
        parts = key.split('.')
        # ['model','layers','{i}','self_attn','sinks']
        layer_idx = int(parts[2])
        tensors = load_file(file_name)
        if key not in tensors:
            continue
        t = tensors[key].to(torch.float32).cpu()
        sinks[layer_idx] = t

    return sinks

