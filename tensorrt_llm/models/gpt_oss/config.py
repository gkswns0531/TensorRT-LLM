from typing import Any, Dict, List, Optional, Union

from tensorrt_llm.models.modeling_utils import PretrainedConfig, QuantConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.layers import MoeConfig


class GptOssConfig(PretrainedConfig):

    def __init__(
        self,
        *,
        layer_types: Optional[List[str]] = None,
        attention_bias: bool = True,
        rope_theta: float = 150000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        initial_context_length: Optional[int] = None,
        num_experts: Optional[int] = None,
        experts_per_token: int = 4,
        sliding_window: Optional[int] = None,
        hidden_act: str = "silu",
        **kwargs,
    ) -> None:
        # gpt-oss specific
        self.layer_types = layer_types or []
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {}
        self.initial_context_length = initial_context_length
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act

        # Build MoE config
        if self.num_experts and self.num_experts > 0:
            moe = MoeConfig(num_experts=self.num_experts,
                            top_k=self.experts_per_token)
        else:
            moe = MoeConfig(num_experts=0, top_k=0)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: Union[str, "transformers.PretrainedConfig"],
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "GptOssConfig":
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf = hf_config_or_dir
        else:
            hf = transformers.AutoConfig.from_pretrained(
                str(hf_config_or_dir), trust_remote_code=kwargs.pop("trust_remote_code", True)
            )

        # HF schema
        num_experts = getattr(hf, "num_local_experts", None)
        experts_per_token = getattr(hf, "num_experts_per_tok", getattr(hf, "experts_per_token", 4))
        rope_scaling = getattr(hf, "rope_scaling", None)
        layer_types = getattr(hf, "layer_types", [])
        attention_bias = getattr(hf, "attention_bias", True)
        initial_context_length = getattr(hf, "initial_context_length", None)

        # Build
        return cls(
            architecture=getattr(hf, "architectures", [""])[0],
            dtype=dtype,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_size=hf.head_dim,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            position_embedding_type="yarn",
            rotary_embedding_dim=None,
            norm_epsilon=getattr(hf, "rms_norm_eps", 1e-5),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
            use_logn_attn=False,
            mapping=mapping,
            quantization=quant_config,
            # gpt-oss specific
            layer_types=layer_types,
            attention_bias=attention_bias,
            rope_theta=getattr(hf, "rope_theta", 150000.0),
            rope_scaling=rope_scaling,
            initial_context_length=initial_context_length,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            sliding_window=getattr(hf, "sliding_window", None),
            hidden_act=getattr(hf, "hidden_act", "silu"),
            **kwargs,
        )

    @classmethod
    def from_original(
        cls,
        original_config: Dict[str, Any],
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "GptOssConfig":
        # Original schema keys
        num_experts = original_config.get("num_experts")
        experts_per_token = original_config.get("experts_per_token", 4)
        rope_theta = float(original_config.get("rope_theta", 150000.0))
        rope_scaling = {
            "factor": original_config.get("rope_scaling_factor", 1.0),
            "beta_fast": original_config.get("rope_ntk_beta", 32.0),
            "beta_slow": original_config.get("rope_ntk_alpha", 1.0),
            "original_max_position_embeddings": original_config.get("initial_context_length", 4096),
            "rope_type": "yarn",
        }

        return cls(
            architecture="GptOssForCausalLM",
            dtype=dtype,
            num_hidden_layers=original_config["num_hidden_layers"],
            num_attention_heads=original_config["num_attention_heads"],
            num_key_value_heads=original_config["num_key_value_heads"],
            head_size=original_config["head_dim"],
            hidden_size=original_config["hidden_size"],
            intermediate_size=original_config["intermediate_size"],
            vocab_size=original_config["vocab_size"],
            max_position_embeddings=original_config.get("max_position_embeddings", 131072),
            position_embedding_type="yarn",
            rotary_embedding_dim=None,
            norm_epsilon=1e-5,
            tie_word_embeddings=False,
            use_logn_attn=False,
            mapping=mapping,
            quantization=quant_config,
            # gpt-oss specific
            layer_types=[],  # original에는 명시 없음 → 모델에서 기본 규칙 보완
            attention_bias=True,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            initial_context_length=original_config.get("initial_context_length", 4096),
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            sliding_window=original_config.get("sliding_window", None),
            hidden_act="silu",
            **kwargs,
        )

    def to_dict(self):
        """Override to handle MoeConfig serialization"""
        output = super().to_dict()
        
        # Handle MoeConfig object - convert to dict if present
        if hasattr(self, 'moe') and self.moe is not None:
            if hasattr(self.moe, 'to_dict'):
                output['moe'] = self.moe.to_dict()
            elif hasattr(self.moe, '__dict__'):
                output['moe'] = {k: v for k, v in self.moe.__dict__.items() 
                               if not k.startswith('_')}
            else:
                # Fallback: try to convert MoeConfig attributes manually
                moe_dict = {}
                for attr in ['num_experts', 'top_k', 'capacity_factor', 'router_aux_loss_coef']:
                    if hasattr(self.moe, attr):
                        moe_dict[attr] = getattr(self.moe, attr)
                if moe_dict:
                    output['moe'] = moe_dict
        
        return output


