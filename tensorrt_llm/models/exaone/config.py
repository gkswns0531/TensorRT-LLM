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
    """Configuration class for Exaone 4.0 model."""

    def __init__(
        self,
        *,
        architecture: str = EXAONE4_ARCHITECTURE,
        # Exaone 4.0 specific parameters
        sliding_window: Optional[int] = 4096,
        sliding_window_pattern: str = "LLLG",
        # YARN scaling parameters
        yarn_factor: float = 1.0,
        yarn_low_freq_factor: float = 1.0,
        yarn_high_freq_factor: float = 4.0,
        yarn_attention_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        # Post-norm configuration
        use_post_norm: bool = True,
        # Missing essential parameters
        hidden_act: str = "silu",
        norm_epsilon: float = 1e-5,
        head_size: Optional[int] = None,
        # Standard parameters
        rotary_base: float = 10000.0,
        rotary_scaling: Optional[dict] = None,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.rope_gpt_neox,
        mapping: Optional[Union[Mapping, dict]] = None,
        **kwargs,
    ):
        # Configure parallel embedding based on tensor parallelism
        use_parallel_embedding = False
        if mapping:
            use_parallel_embedding = mapping.tp_size > 1 if isinstance(
                mapping, Mapping) else mapping.get("tp_size", 1) > 1
        if use_parallel_embedding != kwargs.pop("use_parallel_embedding", None):
            logger.debug(
                f"Using `use_parallel_embedding={use_parallel_embedding}` for Exaone 4.0"
            )

        # Setup YARN scaling if provided
        if rotary_scaling is None and yarn_factor != 1.0:
            rotary_scaling = {
                "type": "yarn",
                "factor": yarn_factor,
                "low_freq_factor": yarn_low_freq_factor,
                "high_freq_factor": yarn_high_freq_factor,
                "attention_factor": yarn_attention_factor,
                "beta_fast": yarn_beta_fast,
                "beta_slow": yarn_beta_slow,
            }

        super().__init__(
            architecture=architecture,
            use_parallel_embedding=use_parallel_embedding,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            attn_bias=attn_bias,
            mlp_bias=mlp_bias,
            position_embedding_type=position_embedding_type,
            mapping=mapping,
            **kwargs,
        )

        # Store Exaone 4.0 specific parameters
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.yarn_factor = yarn_factor
        self.yarn_low_freq_factor = yarn_low_freq_factor
        self.yarn_high_freq_factor = yarn_high_freq_factor
        self.yarn_attention_factor = yarn_attention_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow
        self.use_post_norm = use_post_norm
        self.hidden_act = hidden_act
        self.norm_epsilon = norm_epsilon
        
        # Calculate head_size if not provided
        if head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        else:
            self.head_size = head_size

    def is_sliding_layer(self, layer_idx: int) -> bool:
        """
        Determine if layer uses sliding window attention (static pattern).
        
        Args:
            layer_idx (int): Layer index (0-based)
            
        Returns:
            bool: True if layer uses sliding window, False for global attention
        """
        if not self.sliding_window or not self.sliding_window_pattern:
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
        """
        Create Exaone4Config from HuggingFace config.
        
        Args:
            hf_config_or_dir: HuggingFace config object or model directory
            dtype: Model data type
            mapping: Tensor parallelism mapping
            quant_config: Quantization configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Exaone4Config: TensorRT-LLM configuration for Exaone 4.0
        """
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

        return cls(
            # Model architecture
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            # Model type and dtype
            dtype=dtype,
            # Missing essential fields
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            norm_epsilon=getattr(hf_config, "rms_norm_eps", 1e-5),
            head_size=getattr(hf_config, "head_dim", None),  # HF uses head_dim
            # Exaone 4.0 specific
            sliding_window=getattr(hf_config, "sliding_window", 4096),
            sliding_window_pattern=getattr(hf_config, "sliding_window_pattern", "LLLG"),
            use_post_norm=getattr(hf_config, "use_post_norm", True),
            # RoPE configuration
            rotary_base=getattr(hf_config, "rope_theta", 10000.0),
            rotary_scaling=getattr(hf_config, "rope_scaling", None),
            # YARN parameters
            **yarn_params,
            # System configuration
            mapping=mapping,
            quant_config=quant_config,
            **kwargs
        )
