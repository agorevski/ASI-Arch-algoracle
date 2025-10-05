from argparse import ArgumentParser
import logging
import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record


from trainer import WatermarkTrainer


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_option():
    parser = ArgumentParser(
        'Image and Video watermarking', add_help=False)
    parser.add_argument('--config_file', type=str,
                        default='configs/config.yaml',
                        help='Model config file path')
    parser.add_argument('--dataset_path', type=str,
                        help="Path to the local / AML dataset")
    parser.add_argument("--img_train_path", type=str,
                        default="dalle/train")
    parser.add_argument("--img_test_path", type=str,
                        default="dalle/test")
    parser.add_argument("--video_train_path", type=str,
                        default="sav/sav_train")
    parser.add_argument("--video_test_path", type=str,
                        default="sav/sav_test")
    parser.add_argument("--train_mode", type=str, default="image")
    parser.add_argument("--frame_step", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta_min", type=float, default=1e-4)
    parser.add_argument("--beta_max", type=float, default=40.)
    parser.add_argument("--beta_start_epoch", type=int, default=10)
    parser.add_argument("--beta_epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--noise_start_epoch", type=int, default=50)
    parser.add_argument("--video_start_epoch", type=int, default=80)
    return parser.parse_args()


@record
def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(dist.get_rank())

    args = parse_option()
    wm_model = WatermarkTrainer(args)
    wm_model.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
