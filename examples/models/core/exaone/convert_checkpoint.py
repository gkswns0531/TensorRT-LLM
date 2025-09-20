import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import Exaone4ForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, required=True)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help='Model data type')
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument('--use_weight_only',
                        default=False,
                        action="store_true",
                        help='Quantize weights to INT4/INT8')
    parser.add_argument(
        '--weight_only_precision',
        type=str,
        default='int8',
        choices=['int8', 'int4', 'int4_awq', 'int4_gptq'],
        help='Weight-only precision')
    parser.add_argument('--use_parallel_embedding',
                        action="store_true",
                        default=False,
                        help='Use parallel embedding')
    parser.add_argument('--embedding_sharding_dim',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help='Embedding sharding dimension')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='Number of worker processes')
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()

    if args.model_dir is None:
        logger.error("Model directory must be specified")
        raise RuntimeError("Model directory must be specified")

    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory {args.model_dir} does not exist")
        raise RuntimeError(f"Model directory {args.model_dir} does not exist")

    return args


def convert_checkpoint(args):
    world_size = args.tp_size * args.pp_size
    
    def convert_checkpoint_for_rank(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        model = Exaone4ForCausalLM.from_hugging_face(
            args.model_dir,
            args.dtype,
            mapping=mapping,
            use_parallel_embedding=args.use_parallel_embedding,
            embedding_sharding_dim=args.embedding_sharding_dim,
            **{})
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del model

    if args.workers == 1:
        for rank in range(world_size):
            convert_checkpoint_for_rank(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(convert_checkpoint_for_rank, rank)
                for rank in range(world_size)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    logger.set_level(args.log_level)

    tik = time.time()
    convert_checkpoint(args)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()