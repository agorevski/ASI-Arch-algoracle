from collections import defaultdict
import logging
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch import nn
import torchvision.transforms as transforms
import mlflow
from torch import distributed as dist

import model
import metrics
import utils
import noise
import data.datasets as ds
from loss import WatermarkLoss


mlflow.autolog()
logger = logging.getLogger(__name__)


class WatermarkTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = int(os.environ["LOCAL_RANK"])
        self.cfg = utils.load_configs(args)
        self.model_local = model.InvisMark(self.cfg).to(self.device)
        self.model = DDP(self.model_local, device_ids=[
                         self.device], find_unused_parameters=True)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.loss = WatermarkLoss(self.cfg)
        self.noiser = noise.Noiser(num_transforms=1)
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.cfg['WATERMARK']['ECC_MODE'] == "ecc":
            self.bchecc = model.BCHECC(
                t=self.cfg['WATERMARK']['ECC_T'], m=self.cfg['WATERMARK']['ECC_M'])
            logger.info(f"tot_bits: {self.cfg['WATERMARK']['NUM_BITS']}, data_bytes: {self.bchecc.data_bytes}")

        if self.cfg['train_mode'] == 'video':
            self.train_dataloader = ds.video_train_dataloader(os.path.join(
                args.dataset_path, args.video_train_path),
                frame_step=self.cfg['frame_step'], batch_size=args.batch_size)
            self.eval_dataloader = ds.video_eval_dataloader(os.path.join(
                args.dataset_path, args.video_test_path), frames_per_clip=self.cfg['frame_step'],
                batch_size=1)
        else:
            self.train_dataloader = ds.img_train_dataloader(os.path.join(
                args.dataset_path, args.img_train_path), args.batch_size)
            self.eval_dataloader_list = {}
            for eval_data in args.img_test_path.split(","):
                eval_dataloader = ds.img_eval_dataloader(os.path.join(
                    args.dataset_path, eval_data), 1, num_workers=0)
                self.eval_dataloader_list[eval_data] = eval_dataloader

    def train(self):
        fixed_batch = next(iter(self.train_dataloader))
        for epoch in range(self.cfg['num_epochs']):
            if epoch < self.cfg['warmup_epochs']:
                self._train_one_epoch(epoch, fixed_batch=fixed_batch)
            else:
                self._train_one_epoch(epoch)

            # Only do evaluation on the main process.
            if self.device == 0:
                if self.cfg['train_mode'] == 'video':
                    self._evaluate_video(epoch)
                else:
                    self._evaluate(epoch)
            self.save_ckpt(self.model.state_dict(), epoch, self.output_dir)

    def _train_one_epoch(self, epoch, fixed_batch=None):
        self.model.train()
        for step, batch in enumerate(self.train_dataloader):
            self.opt.zero_grad()
            logger.info(f"Training for epoch: {epoch}, cur_step: {step}")
            if fixed_batch is not None:
                batch = fixed_batch

            # Convert image data shape to be the same as video data [B, F, C, H, W]
            data = batch[0]
            if batch[0].ndim == 4:
                data = batch[0].unsqueeze(1).repeat(
                    1, self.cfg['ENCODER']['NUM_FRAMES'], 1, 1, 1)

            # Generate random bit array for watermarking.
            wm = self._generate_wm(data.shape[0])
            outputs = self.model(data, wm)
            loss_val = self.loss(outputs, wm, epoch)
            loss_val.backward()
            self.opt.step()

    def _generate_wm(self, batch_size):
        if self.cfg['WATERMARK']['ECC_MODE'] == "uuid":
            bits, _ = utils.uuid_to_bits(batch_size)
        elif self.cfg['WATERMARK']['ECC_MODE'] == "ecc":
            assert self.cfg['WATERMARK']['NUM_BITS'] == 256, "Encode 256 bits in ecc mode"
            bits = self.bchecc.batch_encode(batch_size)
        else:
            raise "secret enc_mode is not supported! choose between uuid and ecc."
        return bits[:, :self.cfg['WATERMARK']['NUM_BITS']].to(self.device)

    @torch.no_grad()
    def _calculate_metric(self, imgs, wm_imgs, wm):
        metric = defaultdict(float)
        wm = wm.repeat(imgs.shape[0] // wm.shape[0], 1)
        # Image pixel values should be within [-1, 1], i.e. data_range = 2.0
        metric['psnr'] = metrics.image_psnr(imgs, wm_imgs.cpu())
        metric['ssim'] = metrics.image_ssim(wm_imgs.cpu(), imgs)
        dec_wm = self.model.module.decode(wm_imgs)
        metric['BitAcc'] = metrics.bit_accuracy(wm, dec_wm)
        for key in noise.supported_transforms((256, 256)):
            imgs = self.noiser(wm_imgs, [key])
            dec_wm = self.model.module.decode(imgs)
            metric[f'BitAcc-{key}'] = metrics.bit_accuracy(wm, dec_wm)
            if self.cfg['WATERMARK']['ECC_MODE'] == 'ecc':
                cor_wm = self.bchecc.batch_decode_ecc(dec_wm).cpu()
                metric[f'DataBitAcc-{key}'] = metrics.bit_accuracy(
                    cor_wm[:, :-self.bchecc.bch.ecc_bytes * 8],
                    wm[:, :-self.bchecc.bch.ecc_bytes * 8].cpu())
        return metric

    @torch.inference_mode()
    def _evaluate(self, epoch, num_batches=50):
        self.model.eval()
        for eval_name in self.eval_dataloader_list:
            avg_metrics = defaultdict(float)
            dataloader = self.eval_dataloader_list[eval_name]
            for i, batch in enumerate(dataloader):
                # Convert image data shape to be the same as video data [B, F, C, H, W]
                data = batch[0]
                if batch[0].ndim == 4:
                    data = batch[0].unsqueeze(1).repeat(
                        1, self.cfg['ENCODER']['NUM_FRAMES'], 1, 1, 1)
                wm = self._generate_wm(data.shape[0])
                outputs = self.model(data, wm)
                imgs = data.view(-1, 3, *data.shape[-2:])
                batch_metric = self._calculate_metric(
                    imgs, outputs['final_outputs'], wm)
                for k, v in batch_metric.items():
                    avg_metrics[k] += v
                if i >= num_batches:
                    break
            for k in avg_metrics:
                avg_metrics[k] = avg_metrics[k] / (i + 1.0)
            self._log_metrics(avg_metrics, epoch, f"Eval-{eval_name}")
            self._log_images(imgs, outputs['final_outputs'], epoch, f"Eval-{eval_name}", 1)

    @torch.inference_mode()
    def _evaluate_video(self, epoch, num_batches=5):
        self.model.eval()
        avg_metrics = defaultdict(float)
        for i in range(num_batches):
            batch = next(iter(self.eval_dataloader))[0]
            inputs = torch.cat(
                [batch[:, 0, :, :, :], batch[:, -1, :, :, :]], dim=1)
            wm = self._generate_wm(batch.shape[0])
            out = self.model(inputs, wm)
            residuals = (out['resized_outputs'] - out['resized_inputs'])[0]
            residuals = transforms.Resize(inputs.shape[-2:])(residuals)
            imgs = batch.view(-1, 3, *batch.shape[-2:])
            residuals = residuals.repeat(imgs.shape[0], 1, 1, 1)
            wm_imgs = torch.clamp(imgs.to(self.device)
                                  + residuals, min=-1.0, max=1.0)
            batch_metric = self._calculate_metric(
                imgs, wm_imgs, wm)
            for k, v in batch_metric.items():
                avg_metrics[k] += v
            if i >= num_batches:
                break
        for k in avg_metrics:
            avg_metrics[k] = avg_metrics[k] / (i + 1.0)
        self._log_metrics(avg_metrics, epoch, "Eval")
        self._log_images(imgs, wm_imgs, epoch, "Eval")

    def _log_metrics(self, metrics, epoch, prefix='Eval'):
        for key in metrics:
            try:
                mlflow.log_metric(f"{prefix}/{key}", metrics[key].item(), epoch)
            except Exception as e:
                logger.error(f"Error while logging to AML: {e}")

    def _log_images(self, imgs, wm_imgs, epoch, prefix='Eval', num_samples=1):
        for i in range(num_samples):
            img = transforms.ToPILImage()(
                (imgs[i].cpu() + 1.0)/2.0)
            mlflow.log_image(img, f"{prefix}_inp_epoch_{epoch:04d}_{i:02}.png")
            img = transforms.ToPILImage()(
                (wm_imgs[i].cpu() + 1.0)/2.0)
            mlflow.log_image(img, f"{prefix}_wm_epoch_{epoch:04d}_{i:02}.png")
            img = 20.0 * (imgs[i].cpu() - wm_imgs[i].cpu())
            img = transforms.ToPILImage()((img + 1.0)/2.0)
            mlflow.log_image(img, f"{prefix}_resx20_epoch_{epoch:04d}_{i:02}.png")

    def save_ckpt(self, model_state_dict, epoch, output_dir):
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_state = {
                'model': model_state_dict,
                'epoch': epoch,
            }
            save_path = os.path.join(output_dir, f'ckpt_{epoch}.pth')
            torch.save(save_state, save_path)
            logger.info(f"{save_path} ckpt saved!")
