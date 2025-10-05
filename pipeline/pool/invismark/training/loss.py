import numpy as np
from torch import nn
from focal_frequency_loss import FocalFrequencyLoss as ffl
import lpips
import logging
from collections import defaultdict

from utils import compute_reconstruction_loss


logger = logging.getLogger(__name__)


class WatermarkLoss(nn.Module):
    def __init__(self, cfg):
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
        cur_epoch = min(max(0, epoch - self.cfg['beta_start_epoch']),
                        self.cfg['beta_epochs'] - 1)
        # schedule = np.linspace(
        #     self.cfg['beta_min'], self.cfg['beta_max'], self.cfg['beta_epochs'])
        schedule = np.logspace(np.log10(self.cfg['beta_min']), np.log10(
            self.cfg['beta_max']), self.cfg['beta_epochs'])
        return schedule[cur_epoch]

    def forward(self, out, secret, epoch):
        # Repeat the secret to match the batch size
        secret = secret.repeat(
            out['resized_outputs'].shape[0] // secret.shape[0], 1)
        self.loss['mse'] = compute_reconstruction_loss(
            out['resized_inputs'], out['resized_outputs'], recon_type='yuv').mean()
        self.loss['lpips'] = self.lpips_loss_fn(
            out['resized_inputs'], out['resized_outputs']).mean()
        self.loss['ffl'] = self.ffl_fn(
            out['resized_inputs'], out['resized_outputs']).mean()
        self.loss['bce'] = self.bce_loss_fn(
            out['decode_wm'], secret)
        if epoch >= self.cfg['noise_start_epoch']:
            self.loss['bce'] += self.bce_loss_fn(
                out['decode_wm_noise'], secret)

        return self._beta_coef(epoch) * (self.loss['mse'] + self.loss['lpips'] + self.loss['ffl']) + self.loss['bce']
