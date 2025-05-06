import os
import argparse
import numpy as np

from utils.logger import get_logger

from preprocess.mimic_cxr import preprocess_mimic_cxr

def parse_args():
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--source_dir', required=True, type=str)
    parser.add_argument('--reports_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--test_size', default=0.2, type=float)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--max_samples', type=int, default=None)

    return parser.parse_args()

def main():
    np.random.seed()
    args = parse_args()

    logger = get_logger(args.output_dir, 'data_preparation.log')
    logger.info(f"Arguments: {args}")

    if not os.path.exists(args.source_dir):
        logger.error(f"Can't find source dir: {args.source_dir}")
        return

    if not os.path.exists(args.reports_dir):
        logger.error(f"Can't find reports dir: {args.reports_dir}")
        return

    if not 'mimic-cxr' in args.source_dir.lower() and not 'mimic-cxr' in args.reports_dir.lower():
        logger.error(f"Double-check your dataset paths. src:{args.source_dir} report:{args.reports_dir}")
        return

    train_ratio = 1.0 - args.test_size - args.val_size
    split_ratio = {
        'train': train_ratio,
        'val': args.val_size,
        'test': args.test_size
    }
    logger.info(f"Split ratios: {split_ratio}")

    preprocess_mimic_cxr(logger, args.source_dir, args.reports_dir, args.output_dir, split_ratio, args.seed,
                             args.max_samples)

    logger.info("Data prep finished.")


if __name__ == '__main__':
    main()
