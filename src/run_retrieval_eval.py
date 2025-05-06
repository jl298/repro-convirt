import os
import argparse
import torch
import pandas as pd

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from common.model import ConVIRTModel
from evaluation.retrieval import evaluate_retrieval
from utils.logger import get_logger
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--eval_data_dir', default='./eval_data', type=str)
    parser.add_argument('--img_base_dir', default='./img_base', type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--image2image', action='store_true')
    parser.add_argument('--text2image', action='store_true')

    return parser.parse_args()

def main():
    set_seed()
    args = parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, 'logs')

    logger = get_logger(args.log_dir, 'retrieval_eval.log')
    logger.info(f"Arguments: {args}")

    logger.info(f"Loading model from {args.checkpoint}")
    model = ConVIRTModel()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

    if args.image2image and not args.text2image:
        mode = 'image2image'
    elif args.text2image and not args.image2image:
        mode = 'text2image'
    else:
        mode = 'both'  # Default to both - more is better!

    results = evaluate_retrieval(model, args, mode=mode, logger=logger)

    results_df = pd.DataFrame(results, index=[0])
    logger.info(f"Results: \n{results_df}")

    logger.info(f"Retrieval eval finished.")
    return results

if __name__ == '__main__':
    main()
