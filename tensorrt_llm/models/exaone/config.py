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
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import infer_dtype
from tensorrt_llm.models.modeling_utils import PretrainedConfig, QuantConfig

if TYPE_CHECKING:
    from os import PathLike
    import transformers
    HfConfigOrDir = Union[str, PathLike, transformers.PretrainedConfig]

EXAONE4_ARCHITECTURE = "Exaone4ForCausalLM"


class Exaone4Config(PretrainedConfig):

    def __init__(
        self,
        *,
        architecture: str = EXAONE4_ARCHITECTURE,
        sliding_window: Optional[int] = None,
        sliding_window_pattern: str = "LLLG",
        layer_types: Optional[list] = None,
        yarn_factor: float = 1.0,
        yarn_low_freq_factor: float = 1.0,
        yarn_high_freq_factor: float = 4.0,
        yarn_attention_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        use_post_norm: bool = True,
        use_qk_layernorm: bool = True,
        hidden_act: str = "silu",
        norm_epsilon: float = 1e-5,
        head_size: Optional[int] = None,
        rotary_base: float = 1000000.0,
        rotary_scaling: Optional[dict] = None,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.rope_gpt_neox,
        mapping: Optional[Union[Mapping, dict]] = None,
        **kwargs,
    ):
        # Store Exaone 4.0 specific parameters (Qwen3 pattern)
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.layer_types = layer_types
        self.yarn_factor = yarn_factor
        self.yarn_low_freq_factor = yarn_low_freq_factor
        self.yarn_high_freq_factor = yarn_high_freq_factor
        self.yarn_attention_factor = yarn_attention_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.use_post_norm = use_post_norm
        self.use_qk_layernorm = use_qk_layernorm
        self.hidden_act = hidden_act
        self.norm_epsilon = norm_epsilon
        self.head_size = head_size
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.position_embedding_type = position_embedding_type

        # Setup YARN scaling if provided
        if self.rotary_scaling is None and yarn_factor != 1.0:
            self.rotary_scaling = {
                "type": "yarn",
                "factor": yarn_factor,
                "low_freq_factor": yarn_low_freq_factor,
                "high_freq_factor": yarn_high_freq_factor,
                "attention_factor": yarn_attention_factor,
                "beta_fast": yarn_beta_fast,
                "beta_slow": yarn_beta_slow,
            }

        # Configure parallel embedding based on tensor parallelism
        use_parallel_embedding = False
        if mapping:
            use_parallel_embedding = mapping.tp_size > 1 if isinstance(
                mapping, Mapping) else mapping.get("tp_size", 1) > 1
        
        # Remove conflicting kwargs to prevent "multiple values" error
        kwargs.pop("use_parallel_embedding", None)
        kwargs.pop("qk_layernorm", None)
        kwargs.pop("quant_config", None)
        
        if use_parallel_embedding:
            logger.debug(
                f"Using `use_parallel_embedding={use_parallel_embedding}` for Exaone 4.0"
            )
        
        super().__init__(
            architecture=architecture,
            use_parallel_embedding=use_parallel_embedding,
            qk_layernorm=use_qk_layernorm,
            hidden_act=hidden_act,
            norm_epsilon=norm_epsilon,
            head_size=head_size,
            position_embedding_type=position_embedding_type,
            mapping=mapping,
            **kwargs,
        )

    def to_dict(self):
        output = super().to_dict()
        # Serialize Exaone 4.0 specific fields
        output['sliding_window'] = self.sliding_window
        output['sliding_window_pattern'] = self.sliding_window_pattern
        output['layer_types'] = self.layer_types
        output['yarn_factor'] = self.yarn_factor
        output['yarn_low_freq_factor'] = self.yarn_low_freq_factor
        output['yarn_high_freq_factor'] = self.yarn_high_freq_factor
        output['yarn_attention_factor'] = self.yarn_attention_factor
        output['yarn_beta_fast'] = self.yarn_beta_fast
        output['yarn_beta_slow'] = self.yarn_beta_slow
        output['use_post_norm'] = self.use_post_norm
        output['use_qk_layernorm'] = self.use_qk_layernorm
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output['attn_bias'] = self.attn_bias
        output['mlp_bias'] = self.mlp_bias
        if hasattr(self, 'eos_token_id') and self.eos_token_id is not None:
            output['eos_token_id'] = self.eos_token_id
        if hasattr(self, 'bos_token_id') and self.bos_token_id is not None:
            output['bos_token_id'] = self.bos_token_id
        if hasattr(self, 'pad_token_id') and self.pad_token_id is not None:
            output['pad_token_id'] = self.pad_token_id
        return output

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """
        Determine if layer uses sliding window attention.
        
        Args:
            layer_idx (int): Layer index (0-based)
            
        Returns:
            bool: True if layer uses sliding window, False for global attention
        """
        # Check if sliding window is disabled
        if not self.sliding_window:
            return False
        
        # Use layer_types if available (more precise)
        if hasattr(self, 'layer_types') and self.layer_types and layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "sliding_attention"
        
        # Fallback to pattern-based detection
        if not self.sliding_window_pattern:
            return False
        
        # Last layer is always global attention
        if layer_idx == self.num_hidden_layers - 1:
            return False
        
        # Determine pattern position
        pattern_idx = layer_idx % len(self.sliding_window_pattern)
        return self.sliding_window_pattern[pattern_idx] == "L"

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: "HfConfigOrDir",
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs
    ) -> "Exaone4Config":
        import transformers

        # Load HF config
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = Path(hf_config_or_dir)
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True
            )

        # Infer dtype if auto
        if dtype == "auto":
            dtype = infer_dtype(hf_config, dtype)

        # Extract YARN parameters if available
        yarn_params = {}
        if hasattr(hf_config, 'rope_scaling') and hf_config.rope_scaling:
            scaling_config = hf_config.rope_scaling
            # Support both 'type' and 'rope_type' fields (HF config uses 'rope_type')
            rope_type = scaling_config.get("rope_type", scaling_config.get("type", ""))
            if rope_type in ["yarn", "llama3"]:
                yarn_params = {
                    "yarn_factor": scaling_config.get("factor", 1.0),
                    "yarn_low_freq_factor": scaling_config.get("low_freq_factor", 1.0),
                    "yarn_high_freq_factor": scaling_config.get("high_freq_factor", 4.0),
                    "yarn_attention_factor": scaling_config.get("attention_factor", 1.0),
                    "yarn_beta_fast": scaling_config.get("beta_fast", 32.0),
                    "yarn_beta_slow": scaling_config.get("beta_slow", 1.0),
                }
        
        # Extract special token IDs from HF config (CRITICAL FIX)
        # This fixes the last_token_ids mismatch issue
        special_tokens = {}
        if hasattr(hf_config, 'eos_token_id'):
            special_tokens['eos_token_id'] = hf_config.eos_token_id
        if hasattr(hf_config, 'bos_token_id'):  
            special_tokens['bos_token_id'] = hf_config.bos_token_id
        if hasattr(hf_config, 'pad_token_id'):
            special_tokens['pad_token_id'] = hf_config.pad_token_id

        return cls(
            architecture=hf_config.architectures[0],
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            dtype=dtype,
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            norm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-5),
            head_size=getattr(hf_config, "head_dim", None),
            sliding_window=getattr(hf_config, "sliding_window", None),
            sliding_window_pattern=getattr(hf_config, "sliding_window_pattern", "LLLG"),
            layer_types=getattr(hf_config, "layer_types", None),
            use_post_norm=getattr(hf_config, "use_post_norm", True),
            use_qk_layernorm=getattr(hf_config, "use_qk_layernorm", True),
            rotary_base=getattr(hf_config, "rope_theta", 1000000.0),
            rotary_scaling=getattr(hf_config, "rope_scaling", None),
            attn_bias=getattr(hf_config, "attn_bias", False),
            mlp_bias=getattr(hf_config, "mlp_bias", False),
            **yarn_params,
            **special_tokens,  # CRITICAL FIX: Pass special token IDs
            mapping=mapping,
            quant_config=quant_config,
            **kwargs
        )
