from typing import Optional, Union

import torch

from tensorrt_llm.functional import (LayerNormType, AllReduceFusionOp, constant,
                                     default_net, PositionEmbeddingType, cast,
                                     int32_array)
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
        use_sliding_window = False
        if config.layer_types:
            lt = config.layer_types[layer_idx] if layer_idx < len(
                config.layer_types) else None
            if lt == 'sliding_attention' and getattr(config, 'sliding_window', None):
                use_sliding_window = True

        # Normalize position embedding type for compatibility with plugin expectations
        pos_type = config.position_embedding_type
        if isinstance(pos_type, str):
            if pos_type.lower() == 'yarn':
                pos_type = PositionEmbeddingType.rope_gpt_neox
        # Propagate normalized position embedding type back to config for plugin const params
        self.config.position_embedding_type = pos_type

        self.attention = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=dtype,
            # Always use causal mask type for GPT attention plugin.
            # Sliding window is controlled via host_max_attention_window_sizes.
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attention_bias,
            position_embedding_type=pos_type,
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
        # Ensure attention_params is not None to avoid downstream attribute access errors
        attn_kwargs = dict(attention_mask=attention_mask,
                           use_cache=use_cache,
                           spec_decoding_params=spec_decoding_params,
                           kv_cache_params=kv_cache_params,
                           lora_layer_params=lora_layer_params)
        if attention_params is not None:
            attn_kwargs['attention_params'] = attention_params
        attn_result = self.attention(hidden_states, **attn_kwargs)
        attn_out = attn_result[0] if isinstance(attn_result, tuple) else attn_result

        # Align dtype before residual add
        if hasattr(residual, 'dtype') and hasattr(attn_out, 'dtype') and residual.dtype != attn_out.dtype:
            attn_out = cast(attn_out, residual.dtype)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states, lora_layer_params=lora_layer_params)

        # Align dtype before residual add
        if hasattr(hidden_states, 'dtype') and hasattr(mlp_out, 'dtype') and hidden_states.dtype != mlp_out.dtype:
            mlp_out = cast(mlp_out, hidden_states.dtype)

        hidden_states = residual + mlp_out
        return hidden_states


class _GptOssModel(Module):

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.config = config
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

        # Unify dtype once at model entry to avoid mixed Half/BFloat16 downstream
        target_dtype = getattr(self.config, 'dtype', None)
        if target_dtype is not None and hasattr(hidden_states, 'dtype') and hidden_states.dtype != target_dtype:
            hidden_states = cast(hidden_states, target_dtype)

        # Provide host_max_attention_window_sizes if plugin expects it and not provided by runtime
        kv_params = kwargs.get('kv_cache_params', None)
        if (kv_params is not None and getattr(default_net().plugin_config, 'gpt_attention_plugin', False)
                and getattr(kv_params, 'host_max_attention_window_sizes', None) is None):
            # Build per-PP-layer window sizes: sliding_attention -> config.sliding_window else max_position_embeddings
            window_sizes: list[int] = []
            for layer in self.layers:
                # layer.layer_idx is the global layer index
                window = int(self.config.max_position_embeddings)
                if self.config.layer_types and layer.layer_idx < len(self.config.layer_types):
                    if self.config.layer_types[layer.layer_idx] == 'sliding_attention' and getattr(self.config, 'sliding_window', None):
                        window = int(self.config.sliding_window)
                window_sizes.append(window)
            kv_params.host_max_attention_window_sizes = constant(int32_array(window_sizes))

        # Pass sinks per layer if provided via kwargs; otherwise attempt to use layer.attention.sinks
        sinks_dict: Optional[dict] = kwargs.get('attention_sinks_dict')
        for idx, layer in enumerate(self.layers):
            sinks = None
            if isinstance(sinks_dict, dict):
                sinks = sinks_dict.get(idx)
            if sinks is None and hasattr(layer.attention, 'sinks'):
                param = getattr(layer.attention, 'sinks')
                sinks = getattr(param, 'value', None) or getattr(param, 'data', None) or param
            # Forward required runtime params down to the layer
            layer_kwargs = {}
            for k in (
                'attention_mask',
                'use_cache',
                'spec_decoding_params',
                'kv_cache_params',
                'attention_params',
                'lora_layer_params',
            ):
                if k in kwargs:
                    layer_kwargs[k] = kwargs[k]
            hidden_states = layer(hidden_states, attention_sinks=sinks, **layer_kwargs)
        # apply final norm on last pp rank
        if hasattr(self, 'ln_f'):
            hidden_states = self.ln_f(hidden_states)
        # Return tuple when caching is enabled to satisfy builder expectations
        use_cache = bool(kwargs.get('use_cache', False)) or (kwargs.get('kv_cache_params') is not None)
        if use_cache:
            return hidden_states, None
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


