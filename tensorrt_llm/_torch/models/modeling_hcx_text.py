from .modeling_llama import LlamaForCausalLM
from .modeling_utils import register_auto_model


@register_auto_model("HyperCLOVAXForCausalLM")
class HyperCLOVAXForCausalLM(LlamaForCausalLM):
    pass


