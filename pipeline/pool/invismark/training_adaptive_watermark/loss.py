import numpy as np
import torch
from torch import nn
from focal_frequency_loss import FocalFrequencyLoss as ffl
import lpips
import logging
from collections import defaultdict

from utils import compute_reconstruction_loss


logger = logging.getLogger(__name__)


class WatermarkLoss(nn.Module):
    def __init__(self, cfg):
        """Initialize the WatermarkLoss module.

        Args:
            cfg: Configuration dictionary containing loss parameters including
                'beta_start_epoch', 'beta_epochs', 'beta_min', 'beta_max',
                and 'noise_start_epoch'.
        """
        super(WatermarkLoss, self).__init__()
        self.cfg = cfg
        # lpips: image should be RGB, IMPORTANT: normalized to [-1,1]
        self.lpips_loss_fn = lpips.LPIPS(net='vgg')
        self.lpips_loss_fn.cuda()
        self.ffl_fn = ffl(loss_weight=1.0, alpha=1.0)
        self.bce_loss_fn = nn.BCELoss()
        self.loss = defaultdict(float)
        self.train_bit_accuracy = defaultdict(float)

    def _beta_coef(self, epoch):
        """Calculate the beta coefficient for the current epoch.

        Uses a logarithmic schedule to interpolate between beta_min and beta_max
        based on the current epoch relative to the beta start epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            float: The beta coefficient value for weighting the reconstruction loss.
        """
        cur_epoch = min(max(0, epoch - self.cfg['beta_start_epoch']),
                        self.cfg['beta_epochs'] - 1)
        # schedule = np.linspace(
        #     self.cfg['beta_min'], self.cfg['beta_max'], self.cfg['beta_epochs'])
        schedule = np.logspace(np.log10(self.cfg['beta_min']), np.log10(
            self.cfg['beta_max']), self.cfg['beta_epochs'])
        return schedule[cur_epoch]

    def forward(self, out, secret, epoch):
        """Compute the total watermarking loss.

        Combines reconstruction losses (MSE, LPIPS, FFL) with watermark decoding
        loss (BCE). After the noise start epoch, applies difficulty-weighted BCE
        loss to focus training on problematic bits.

        Args:
            out: Dictionary containing model outputs with keys:
                - 'resized_inputs': Original input images.
                - 'resized_outputs': Watermarked output images.
                - 'decode_wm': Decoded watermark from clean images.
                - 'decode_wm_noise': Decoded watermark from noised images.
            secret: Ground truth watermark tensor.
            epoch: Current training epoch.

        Returns:
            torch.Tensor: Total weighted loss combining reconstruction and
                decoding losses.
        """
        # Repeat the secret to match the batch size
        secret = secret.repeat(
            out['resized_outputs'].shape[0] // secret.shape[0], 1)
        self.loss['mse'] = compute_reconstruction_loss(
            out['resized_inputs'], out['resized_outputs'], recon_type='yuv').mean()
        self.loss['lpips'] = self.lpips_loss_fn(
            out['resized_inputs'], out['resized_outputs']).mean()
        self.loss['ffl'] = self.ffl_fn(
            out['resized_inputs'], out['resized_outputs']).mean()
        
        # Standard BCE loss for clean decoding
        self.loss['bce'] = self.bce_loss_fn(out['decode_wm'], secret)
        
        # Enhanced loss for noised outputs with per-bit weighting
        # This helps reduce decode errors by focusing on difficult bits
        if epoch >= self.cfg['noise_start_epoch']:
            # Calculate per-bit difficulty (higher error = more difficult)
            bit_errors = (out['decode_wm_noise'] - secret).abs()
            
            # Focus training on difficult bits (helps decode errors)
            # But don't increase overall loss magnitude (preserves PSNR/SSIM)
            difficulty_weights = 1.0 + bit_errors.detach()  # Range: [1.0, 2.0]
            difficulty_weights = difficulty_weights / difficulty_weights.mean()  # Normalize
            
            # Standard BCE for noised output
            bce_noise = self.bce_loss_fn(out['decode_wm_noise'], secret)
            
            # Weighted BCE that focuses on problematic bits
            weighted_bce = (torch.nn.functional.binary_cross_entropy(
                out['decode_wm_noise'], secret, reduction='none'
            ) * difficulty_weights).mean()
            
            # Blend standard and weighted BCE (70% standard, 30% weighted)
            self.loss['bce'] += 0.7 * bce_noise + 0.3 * weighted_bce

        return self._beta_coef(epoch) * (self.loss['mse'] + self.loss['lpips'] + self.loss['ffl']) + self.loss['bce']
