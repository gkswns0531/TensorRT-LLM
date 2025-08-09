from typing import Optional, Union

import torch

from ...layers import Embedding
from ...module import Module
from ..modeling_utils import DecoderLayerList, DecoderModelForCausalLM
from .config import GptOssConfig


class _GptOssDecoderLayer(Module):

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        # NOTE: Detailed attention/MoE wiring will be implemented in subsequent edits.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


