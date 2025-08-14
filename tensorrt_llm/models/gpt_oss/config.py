from typing import Any, Dict, List, Optional, Union
import json

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
        moe: Optional[Union[MoeConfig, dict]] = None,
        experts_per_token: int = 4,
        sliding_window: Optional[int] = None,
        hidden_act: str = "silu",
        **kwargs,
    ) -> None:
        self.layer_types = layer_types or []
        self.attention_bias = attention_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {}
        self.initial_context_length = initial_context_length
        self.experts_per_token = experts_per_token
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act

        # Build MoE config (same as Qwen approach)  
        if moe is None:
            moe = MoeConfig(num_experts=0, top_k=0)
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        if 'logits_dtype' not in kwargs:
            kwargs['logits_dtype'] = 'float32'
        super().__init__(hidden_act=hidden_act, **kwargs)

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir,
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

        num_experts = getattr(hf, "num_local_experts", None)
        experts_per_token = getattr(hf, "num_experts_per_tok", getattr(hf, "experts_per_token", 4))
        rope_scaling = getattr(hf, "rope_scaling", None)
        layer_types = getattr(hf, "layer_types", [])
        attention_bias = getattr(hf, "attention_bias", True)
        initial_context_length = getattr(hf, "initial_context_length", None)

        moe_cfg = None
        if num_experts is not None and num_experts > 0:
            moe_cfg = MoeConfig(num_experts=num_experts, top_k=experts_per_token).validate()

        hidden_act = getattr(hf, "hidden_act", "silu")
        if moe_cfg is not None and moe_cfg.num_experts > 0:
            hidden_act = "swiglu"

        # Derive head_size with torch-backend-compatible fallback
        hidden_size = hf.hidden_size
        num_attention_heads = hf.num_attention_heads
        config_head_dim = getattr(hf, 'head_dim', None)
        calculated_head_dim = hidden_size // num_attention_heads
        
        # CRITICAL FIX: Always use calculated head_dim for GPT-OSS models
        # The config.json head_dim (64) is inconsistent with actual dimensions (45)
        if config_head_dim != calculated_head_dim:
            print(f"[GPT-OSS] Config head_dim={config_head_dim}, but calculated head_dim={calculated_head_dim}")
            print(f"[GPT-OSS] Using calculated value for TensorRT-LLM compatibility")
        
        head_dim = calculated_head_dim
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Invalid attention dims: hidden_size={hidden_size}, num_attention_heads={num_attention_heads}."
            )

        # Validate layer_types length when provided
        if isinstance(layer_types, list) and len(layer_types) > 0:
            if len(layer_types) != hf.num_hidden_layers:
                raise ValueError(
                    f"layer_types length {len(layer_types)} must equal num_hidden_layers {hf.num_hidden_layers}"
                )

        return cls(
            architecture=getattr(hf, "architectures", [""])[0],
            dtype=dtype,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_size=head_dim,
            hidden_size=hidden_size,
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            max_position_embeddings=hf.max_position_embeddings,
            position_embedding_type="yarn",
            rotary_embedding_dim=head_dim,
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
            moe=moe_cfg,
            experts_per_token=experts_per_token,
            sliding_window=getattr(hf, "sliding_window", None),
            hidden_act=hidden_act,
            logits_dtype=dtype,
            **kwargs,
        )

    def to_dict(self):
        output = super().to_dict()
        output['moe'] = self.moe.to_dict()
        return output


