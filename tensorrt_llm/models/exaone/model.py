# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import TYPE_CHECKING, Any, Dict, Optional

from tensorrt_llm.models.exaone.convert import (QuantizeModifiers, Weights,
                                                load_exaone4_weights_from_hf_model,
                                                non_modelopt_quantize_if_needed)
from tensorrt_llm.quantization.mode import (MODELOPT_FLOW_QUANTIZATIONS,
                                            QuantAlgo)

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (AllReduceFusionOp, AllReduceParams, LayerNormType,
                           Tensor, cast, recv, send)
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       ColumnLinear, Embedding, GatedMLP, KeyValueCacheParams,
                       LoraParams, PositionEmbeddingType, RmsNorm)
from ...lora_helper import LoraConfig, use_lora
from ...mapping import Mapping
from ...module import Module
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, save_checkpoint, save_config)
from .config import Exaone4Config

if TYPE_CHECKING:
    from .config import HfConfigOrDir


class Exaone4DecoderLayer(Module):

    def __init__(self, config: Exaone4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Determine if this layer uses sliding window (static pattern)
        self.is_sliding = config.is_sliding_layer(layer_idx)
        
        # Parallelization setup
        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.local_layer_idx = layer_idx - layers_range[0]

        qk_layernorm = config.use_qk_layernorm

        self.attention = Attention(
            local_layer_idx=self.local_layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attention_head_size=config.head_size,
            qk_layernorm=qk_layernorm,
            layernorm_type=LayerNormType.RmsNorm if qk_layernorm else LayerNormType.LayerNorm,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.sliding_window_causal if self.is_sliding else AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            is_local=self.is_sliding,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )

        # Initialize MLP (same as standard)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=config.mlp_bias,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )

        # Post-normalization layers (Exaone 4.0 specific)
        if config.use_post_norm:
            self.post_attention_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype,
            )
            
            self.post_feedforward_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype,
            )
        else:
            # Fallback to standard pre-normalization
            self.input_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype,
            )
            
            self.post_attention_layernorm = RmsNorm(
                normalized_shape=config.hidden_size,
                eps=config.norm_epsilon,
                dtype=config.dtype,
            )

    def forward(self,
                hidden_states,
                attention_mask=None,
                use_cache=False,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                lora_layer_params=None):
        """
        Forward pass with Post-norm architecture support
        """
        if self.config.use_post_norm:
            return self._forward_post_norm(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                spec_decoding_params=spec_decoding_params,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
                lora_layer_params=lora_layer_params,
            )
        else:
            return self._forward_pre_norm(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                spec_decoding_params=spec_decoding_params,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
                lora_layer_params=lora_layer_params,
            )

    def _forward_post_norm(self,
                           hidden_states: Tensor,
                           attention_mask=None,
                           use_cache=False,
                           spec_decoding_params=None,
                           kv_cache_params=None,
                           attention_params=None,
                           lora_layer_params=None):
        """
        Post-normalization forward pass (Exaone 4.0 specific)
        Architecture: Input → Attention → Norm → Residual → MLP → Norm → Residual
        """
        # === Attention Block ===
        residual = hidden_states
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
        )
        
        if use_cache:
            attention_output, presents = attention_output
        
        # Post-attention normalization + residual
        attention_output = self.post_attention_layernorm(attention_output)
        hidden_states = residual + attention_output

        # === MLP Block ===
        residual = hidden_states
        mlp_output = self.mlp(
            hidden_states=hidden_states,
            lora_layer_params=lora_layer_params,
        )
        
        # Post-feedforward normalization + residual
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        if use_cache:
            return (hidden_states, presents)
        return hidden_states

    def _forward_pre_norm(self,
                          hidden_states: Tensor,
                          attention_mask=None,
                          use_cache=False,
                          spec_decoding_params=None,
                          kv_cache_params=None,
                          attention_params=None,
                          lora_layer_params=None):
        """
        Standard pre-normalization forward pass (fallback)
        """
        # === Attention Block ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_layer_params=lora_layer_params,
        )
        
        if use_cache:
            attention_output, presents = attention_output
        
        hidden_states = residual + attention_output

        # === MLP Block ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(
            hidden_states=hidden_states,
            lora_layer_params=lora_layer_params,
        )
        hidden_states = residual + mlp_output

        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class Exaone4Model(Module):

    def __init__(self, config: Exaone4Config):
        super().__init__()
        
        # Vocabulary embedding  
        self.vocab_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            sharding_dim=config.embedding_sharding_dim,
        )

        # Decoder layers with Exaone 4.0 specific architecture
        self.layers = DecoderLayerList(
            Exaone4DecoderLayer,
            config
        )

        # Final layer normalization
        self.ln_f = RmsNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype,
        )

    def forward(self,
                input_ids,
                position_ids,
                use_cache=False,
                attention_mask=None,
                spec_decoding_params=None,
                kv_cache_params=None,
                attention_params=None,
                hidden_states=None,
                all_reduce_workspace=None,
                lora_params=None):

        if hidden_states is None:
            hidden_states = self.vocab_embedding(input_ids)

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            spec_decoding_params=spec_decoding_params,
            lora_params=lora_params,
        )

        if use_cache:
            hidden_states, presents = hidden_states

        hidden_states = self.ln_f(hidden_states)

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class Exaone4ForCausalLM(DecoderModelForCausalLM):
    config_class = Exaone4Config
    
    def __init__(self, config: Exaone4Config):
        transformer = Exaone4Model(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, 
                                           config.mapping.tp_size)
        
        lm_head = ColumnLinear(
            config.hidden_size,
            vocab_size_padded,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )
        
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir: "HfConfigOrDir",
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs
    ) -> "Exaone4ForCausalLM":
        """
        Create Exaone4ForCausalLM from HuggingFace model or checkpoint directory.
        """
        config = Exaone4Config.from_hugging_face(
            hf_model_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs
        )
        
        model = cls(config)
        
        # Load weights if model directory provided
        if isinstance(hf_model_or_dir, str):
            weights = load_exaone4_weights_from_hf_model(
                hf_model_or_dir, config, model
            )
            model.load(weights)
        
        return model

    def check_config(self, config: Exaone4Config):
        """Validate Exaone 4.0 specific configuration"""
        config.set_if_not_exist('apply_query_key_layer_scaling', False)
        config.set_if_not_exist('attention_head_size', config.head_size)
        config.set_if_not_exist('kv_channels', config.head_size)
        
        # Validate sliding window pattern
        if config.sliding_window_pattern and len(config.sliding_window_pattern) == 0:
            raise ValueError("sliding_window_pattern cannot be empty")
        
        # Validate YARN parameters
        if config.yarn_factor < 1.0:
            raise ValueError("yarn_factor must be >= 1.0")

    def use_lora(self, lora_config: LoraConfig):
        """Enable LoRA for Exaone 4.0 model"""
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
