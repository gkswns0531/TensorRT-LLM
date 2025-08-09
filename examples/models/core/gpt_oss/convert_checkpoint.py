import argparse
import json
import os
from pathlib import Path

from tensorrt_llm.logger import logger
from tensorrt_llm.models.gpt_oss.config import GptOssConfig
from tensorrt_llm.models.gpt_oss.convert import convert_and_save


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
    return parser.parse_args()


def try_load_original_config(model_dir: Path):
    orig_cfg = model_dir / 'original' / 'config.json'
    if orig_cfg.exists():
        with open(orig_cfg, 'r') as f:
            return json.load(f)
    return None


def main():
    args = parse_arguments()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cfg = try_load_original_config(model_dir)
    if original_cfg is not None:
        logger.info('Loading original gpt-oss config.json')
        cfg = GptOssConfig.from_original(original_cfg, dtype=args.dtype,
                                         mapping=None, quant_config=None)
    else:
        logger.info('Loading HF gpt-oss config.json')
        cfg = GptOssConfig.from_hugging_face(str(model_dir), dtype=args.dtype,
                                             mapping=None, quant_config=None,
                                             trust_remote_code=True)

    # Save config and placeholder shards (full conversion will be added incrementally)
    convert_and_save(model_dir, output_dir, cfg)
    logger.info(f'Prepared TensorRT-LLM checkpoint at {output_dir}')


if __name__ == '__main__':
    main()


