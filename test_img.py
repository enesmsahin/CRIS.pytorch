import argparse
import os
import warnings
from pathlib import Path
import cv2
import torch
import torch.nn.parallel
import torch.utils.data
from loguru import logger

import utils.config as config
from engine.engine import inference_single
from model import build_segmenter
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Text input")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model pth")
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg, args


@logger.catch
def main():
    args, args_2 = get_parser()
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=0,
                 filename="test.log",
                 mode="a")
    logger.info(args)

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    logger.info(model)

    logger.info(args)
    if os.path.isfile(args_2.model):
        logger.info("=> loading checkpoint '{}'".format(args_2.model))
        checkpoint = torch.load(args_2.model)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args_2.model))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args_2.model))

    # inference
    pred = inference_single(args_2.img, args_2.text, model, args)

    save_path = str(Path(args.output_dir) / (Path(args_2.img).stem + ".png"))
    cv2.imwrite(save_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0]) 
    logger.info(f"Output image is saved to {save_path}")

if __name__ == '__main__':
    main()
