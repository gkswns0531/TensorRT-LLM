from typing import Optional, Union

import torch

from ...functional import LayerNormType
from ...layers import (Attention, AttentionMaskType, Embedding, GatedMLP,
                       RmsNorm)
from ...module import Module
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM
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
            attention_window_size=sliding_window,
        )

        # For initial milestone, use GatedMLP; MoE wiring will be added later.
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.attention(hidden_states)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
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

    def forward(self,
                input_ids: torch.Tensor,
                hidden_states: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids)

        hidden_states = self.layers.forward(hidden_states)
        return hidden_states


class GptOssForCausalLM(DecoderModelForCausalLM):
    config_class = GptOssConfig

    def __init__(self, config: GptOssConfig):
        transformer = _GptOssModel(config)
        super().__init__(config, transformer, lm_head=None)

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir: Union[str, "transformers.PreTrainedModel"],
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


