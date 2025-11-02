import argparse
import glob
import json
import logging
import os
import struct
import uuid
from collections import OrderedDict

import numpy as np
import torch
import yaml
from kornia.color import rgb_to_yuv
from torch import distributed as dist


def load_configs(args: argparse.Namespace) -> dict:
    with open(args.config_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    # Update config value with input args.
    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value
    # # Update config value with derived values.
    # if config_dict['train_mode'] == 'image':
    #     config_dict['ENCODER']['NUM_FRAMES'] = 1
    print("Loaded config:", config_dict)
    return config_dict


def compute_reconstruction_loss(
        inputs,
        reconstructions,
        recon_type='rgb'):
    if recon_type == 'rgb':
        rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1, 2, 3])
    elif recon_type == 'yuv':
        reconstructions_yuv = rgb_to_yuv((reconstructions + 1) / 2)
        inputs_yuv = rgb_to_yuv((inputs + 1) / 2)
        yuv_loss = torch.mean(
            (reconstructions_yuv - inputs_yuv)**2, dim=[2, 3])
        yuv_scale = torch.tensor([1, 100, 100]).unsqueeze(
            1).float().to(yuv_loss.device)  # [3,1]
        rec_loss = torch.mm(yuv_loss, yuv_scale).squeeze(1)
    else:
        raise ValueError(f"Unknown recon type {recon_type}")
    return rec_loss


def uuid_to_bits(batch_size):
    uid = [uuid.uuid4() for _ in range(batch_size)]
    seq = np.array([[n for n in u.bytes] for u in uid], dtype=np.uint8)
    bits = torch.Tensor(np.unpackbits(seq, axis=1)).to(torch.float32)
    strs = [str(u) for u in uid]
    return bits, strs


def uuid_to_bytes(batch_size):
    return [uuid.uuid4().bytes for _ in range(batch_size)]


def bits_to_uuid(bits, threshold=0.5):
    bits = np.array(bits) >= threshold
    nums = np.packbits(bits.astype(np.int64), axis=-1)
    res = []
    for j in range(nums.shape[0]):
        bstr = b''
        for i in range(nums.shape[1]):
            bstr += struct.pack('>B', nums[j][i])
        res.append(str(uuid.UUID(bytes=bstr)))
    return res


def save_ckpt(model_state_dict, epoch, output_dir) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        save_state = {
            'model': model_state_dict,
            'epoch': epoch,
        }
        save_path = os.path.join(output_dir, f'ckpt_{epoch}.pth')
        torch.save(save_state, save_path)
        logging.info(f"{save_path} ckpt saved!")

def save_metrics(metrics, epoch, output_dir) -> None:
    # Convert tensor values to Python floats for JSON serialization
    serializable_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    save_path = os.path.join(output_dir, f'metrics_{epoch}.json')
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4, sort_keys=True)
    logging.info(f"{save_path} metrics saved!")


def output_all_metrics(output_dir) -> None:
    # Find all metrics JSON files
    metrics_files = sorted(glob.glob(os.path.join(output_dir, 'metrics_*.json')))
    if not metrics_files:
        logging.info(f"No metrics files found in {output_dir}")
        return

    # Load all metrics into a dictionary indexed by epoch
    all_metrics = OrderedDict()
    for filepath in metrics_files:
        try:
            epoch = int(os.path.basename(filepath).replace('metrics_', '').replace('.json', ''))
            with open(filepath, 'r') as f:
                all_metrics[epoch] = json.load(f)
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")

    if not all_metrics:
        logging.info("No valid metrics loaded")
        return

    # Get all unique metric names
    all_metric_names = set()
    for metrics in all_metrics.values():
        all_metric_names.update(metrics.keys())
    metric_names = sorted(all_metric_names)

    # Build formatted table
    epochs = sorted(all_metrics.keys())

    # Calculate column widths
    metric_col_width = max(len(name) for name in metric_names) + 2
    epoch_col_width = 12  # Width for numeric values with 4 decimal places

    # Prepare output string
    output_lines = []

    # Header row
    header = f"{'Metric':<{metric_col_width}}"
    for epoch in epochs:
        header += f"{'Epoch ' + str(epoch):>{epoch_col_width}}"
    output_lines.append("\n" + header)
    output_lines.append("-" * (metric_col_width + epoch_col_width * len(epochs)))

    # Data rows
    for metric_name in metric_names:
        row = f"{metric_name:<{metric_col_width}}"
        for epoch in epochs:
            value = all_metrics[epoch].get(metric_name, "N/A")
            if isinstance(value, float):
                row += f"{value:>{epoch_col_width}.4f}"
            else:
                row += f"{str(value):>{epoch_col_width}}"
        output_lines.append(row)

    output_lines.append("")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to file
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write('\n'.join(output_lines))
    logging.info(f"Results saved to {results_path}")
