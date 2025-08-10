from typing import Optional, Union

import torch

from tensorrt_llm.functional import LayerNormType, AllReduceFusionOp, constant, default_net
from tensorrt_llm.layers import (Attention, AttentionMaskType, ColumnLinear, Embedding,
                         GatedMLP, RmsNorm, MOE, MoeConfig)
from tensorrt_llm.parameter import Parameter
from tensorrt_llm.module import Module
from tensorrt_llm.models.modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from .config import GptOssConfig


class _GptOssDecoderLayer(Module):

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        # sliding window control via layer_types if provided
        sliding_window = None
        if config.layer_types:
            lt = config.layer_types[layer_idx] if layer_idx < len(
                config.layer_types) else None
            if lt == 'sliding_attention':
                sliding_window = getattr(config, 'sliding_window', None)

        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attention_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rope_theta,
            rotary_embedding_scaling=config.rope_scaling,
            tp_rank=config.mapping.tp_rank,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
            layernorm_type=LayerNormType.RmsNorm,
        )
        # register sinks parameter under attention for weight loading compatibility
        setattr(self.attention, 'sinks', Parameter(shape=(config.num_attention_heads // max(1, config.mapping.tp_size), ),
                                                       dtype='float32'))

        # Use MOE when configured; fallback to GatedMLP otherwise (same as Qwen)
        if config.moe.has_moe():
            self.mlp = MOE(moe_config=config.moe,
                           hidden_size=config.hidden_size,
                           ffn_hidden_size=config.intermediate_size,
                           hidden_act=config.hidden_act,
                           mapping=config.mapping,
                           bias=True,
                           dtype=dtype,
                           tp_size=config.mapping.tp_size,
                           tp_group=config.mapping.tp_group,
                           quant_mode=config.quant_mode)
        else:
            self.mlp = GatedMLP(hidden_size=config.hidden_size,
                                ffn_hidden_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                dtype=dtype,
                                bias=True,
                                tp_group=config.mapping.tp_group,
                                tp_size=config.mapping.tp_size,
                                quant_mode=config.quant_mode)

        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

    def forward(self, 
                hidden_states: torch.Tensor, 
                *, 
                attention_sinks: Optional[torch.Tensor] = None,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None) -> torch.Tensor:
        
        # Basic NVFP4 compatibility check (similar to Llama)
        if (default_net().plugin_config.reduce_fusion 
            and default_net().plugin_config.user_buffer 
            and self.config.quant_mode.has_nvfp4()):
            assert default_net().plugin_config.gemm_plugin == "nvfp4", \
                "UB with nvfp4 model must use nvfp4 gemm plugin"
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # try to pass sinks if attention supports it; otherwise fallback
        attn_out = None
        sinks_arg = attention_sinks
        if sinks_arg is None and hasattr(self.attention, 'sinks'):
            sinks_tensor = getattr(self.attention, 'sinks')
            # Parameter may carry .value or .data depending on backend; forward raw to attention
            sinks_arg = getattr(sinks_tensor, 'value', None) or getattr(sinks_tensor, 'data', None) or sinks_tensor
        attn_out = self.attention(hidden_states, 
                                attention_mask=attention_mask,
                                use_cache=use_cache,
                                spec_decoding_params=spec_decoding_params,
                                kv_cache_params=kv_cache_params,
                                attention_params=attention_params,
                                lora_layer_params=lora_layer_params,
                                attention_sinks=sinks_arg)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states, lora_layer_params=lora_layer_params)
        hidden_states = residual + mlp_out
        return hidden_states


class _GptOssModel(Module):

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(_GptOssDecoderLayer, config)

        # final layer norm
        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: torch.Tensor,
                hidden_states: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)

        # Pass sinks per layer if provided via kwargs; otherwise attempt to use layer.attention.sinks
        sinks_dict: Optional[dict] = kwargs.get('attention_sinks_dict')
        for idx, layer in enumerate(self.layers):
            sinks = None
            if isinstance(sinks_dict, dict):
                sinks = sinks_dict.get(idx)
            if sinks is None and hasattr(layer.attention, 'sinks'):
                param = getattr(layer.attention, 'sinks')
                sinks = getattr(param, 'value', None) or getattr(param, 'data', None) or param
            hidden_states = layer(hidden_states, attention_sinks=sinks)
        # apply final norm on last pp rank
        if hasattr(self, 'ln_f'):
            hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GptOssForCausalLM(DecoderModelForCausalLM):
    config_class = GptOssConfig

    def __init__(self, config: GptOssConfig):
        transformer = _GptOssModel(config)
        vocab_size_padded = ((config.vocab_size + config.mapping.tp_size - 1)
                             // config.mapping.tp_size) * config.mapping.tp_size if config.mapping and config.mapping.tp_size > 0 else config.vocab_size

        if config.mapping.is_last_pp_rank():
            lm_head = ColumnLinear(config.hidden_size,
                                   vocab_size_padded,
                                   bias=False,
                                   dtype=config.dtype,
                                   tp_group=config.mapping.tp_group,
                                   tp_size=config.mapping.tp_size,
                                   gather_output=True)
        else:
            lm_head = None

        super().__init__(config, transformer, lm_head)

        # Customize weight loader mapping for MoE to match gpt-oss converter keys
        for module in self.transformer.layers:
            if hasattr(module.mlp, 'fc'):
                # Update only the necessary entries to avoid clobbering NVFP4 mappings
                mapping_dict = getattr(module.mlp.fc, 'tllm_to_externel_key_dict', None)
                if isinstance(mapping_dict, dict):
                    mapping_dict.update({
                        "weight": "mlp.fc.weight",
                        "bias": "mlp.fc.bias",
                    })
                else:
                    module.mlp.fc.tllm_to_externel_key_dict = {
                        "weight": "mlp.fc.weight",
                        "bias": "mlp.fc.bias",
                    }
            if hasattr(module.mlp, 'proj'):
                # Update only the necessary entries to avoid clobbering NVFP4 mappings
                mapping_dict = getattr(module.mlp.proj, 'tllm_to_externel_key_dict', None)
                if isinstance(mapping_dict, dict):
                    mapping_dict.update({
                        "weight": "mlp.proj.weight",
                        "bias": "mlp.proj.bias",
                    })
                else:
                    module.mlp.proj.tllm_to_externel_key_dict = {
                        "weight": "mlp.proj.weight",
                        "bias": "mlp.proj.bias",
                    }
            if hasattr(module.mlp, 'router'):
                module.mlp.router.tllm_to_externel_key_dict = {
                    "mlp": "mlp",
                    "router": "mlp.router"
                }

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir,
        dtype: str = "auto",
        mapping=None,
        quant_config=None,
        **kwargs,
    ):
        config = GptOssConfig.from_hugging_face(hf_model_or_dir,
                                                dtype=dtype,
                                                mapping=mapping,
                                                quant_config=quant_config,
                                                **kwargs)
        return cls(config)


