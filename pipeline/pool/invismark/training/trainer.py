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
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
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

            # Evaluation is now distributed across all GPUs
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
        
        # Reset decode error counter at the start of metric calculation
        if self.cfg['WATERMARK']['ECC_MODE'] == 'ecc':
            self.bchecc.reset_error_count()
        
        # Image pixel values should be within [-1, 1], i.e. data_range = 2.0
        metric['psnr'] = metrics.image_psnr(imgs, wm_imgs.cpu())
        metric['ssim'] = metrics.image_ssim(wm_imgs.cpu(), imgs)
        dec_wm = self.model.module.decode(wm_imgs)
        metric['BitAcc'] = metrics.bit_accuracy(wm, dec_wm)

        # Parallelize noise transformation processing via batching
        transform_keys = list(noise.supported_transforms((256, 256)).keys())
        all_transformed = []
        
        # Apply all noise transformations and collect results
        for key in transform_keys:
            transformed_imgs = self.noiser(wm_imgs, [key])
            all_transformed.append(transformed_imgs)
        
        # Batch decode all transformed images at once for GPU efficiency
        batched_imgs = torch.cat(all_transformed, dim=0)
        batched_dec_wm = self.model.module.decode(batched_imgs)
        
        # Split results and calculate metrics for each transformation
        batch_size = wm_imgs.shape[0]
        for idx, key in enumerate(transform_keys):
            start_idx = idx * batch_size
            end_idx = (idx + 1) * batch_size
            dec_wm = batched_dec_wm[start_idx:end_idx]
            
            metric[f'BitAcc-{key}'] = metrics.bit_accuracy(wm, dec_wm)
            
            if self.cfg['WATERMARK']['ECC_MODE'] == 'ecc':
                cor_wm = self.bchecc.batch_decode_ecc(dec_wm).cpu()
                metric[f'DataBitAcc-{key}'] = metrics.bit_accuracy(
                    cor_wm[:, :-self.bchecc.bch.ecc_bytes * 8],
                    wm[:, :-self.bchecc.bch.ecc_bytes * 8].cpu())
        
        # Add decode error count to metrics
        if self.cfg['WATERMARK']['ECC_MODE'] == 'ecc':
            metric['decode_errors'] = self.bchecc.get_error_count()
        
        return metric

    @torch.inference_mode()
    def _evaluate(self, epoch, num_batches=50):
        self.model.eval()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        for eval_name in self.eval_dataloader_list:
            local_metrics = defaultdict(float)
            dataloader = self.eval_dataloader_list[eval_name]
            
            # Distribute batches across GPUs
            batch_count = 0
            last_imgs = None
            last_outputs = None
            
            for i, batch in enumerate(dataloader):
                # Skip batches not assigned to this GPU
                if i % world_size != rank:
                    continue
                    
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
                    local_metrics[k] += v
                batch_count += 1
                
                # Store last processed batch for logging
                if rank == 0:
                    last_imgs = imgs
                    last_outputs = outputs['final_outputs']
                
                if i >= num_batches:
                    break
            
            # Aggregate metrics across all GPUs
            avg_metrics = self._aggregate_metrics(local_metrics, batch_count)
            
            # Only rank 0 logs metrics and images
            if rank == 0 and last_imgs is not None:
                self._log_metrics(avg_metrics, epoch, f"Eval-{eval_name}")
                self._log_images(last_imgs, last_outputs, epoch, f"Eval-{eval_name}", 1)

    @torch.inference_mode()
    def _evaluate_video(self, epoch, num_batches=5):
        self.model.eval()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        local_metrics = defaultdict(float)
        batch_count = 0
        last_imgs = None
        last_wm_imgs = None
        
        for i in range(num_batches):
            # Skip batches not assigned to this GPU
            if i % world_size != rank:
                continue
                
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
                local_metrics[k] += v
            batch_count += 1
            
            # Store last processed batch for logging
            if rank == 0:
                last_imgs = imgs
                last_wm_imgs = wm_imgs
        
        # Aggregate metrics across all GPUs
        avg_metrics = self._aggregate_metrics(local_metrics, batch_count)
        
        # Only rank 0 logs metrics and images
        if rank == 0 and last_imgs is not None:
            self._log_metrics(avg_metrics, epoch, "Eval")
            self._log_images(last_imgs, last_wm_imgs, epoch, "Eval")

    def _aggregate_metrics(self, local_metrics, batch_count):
        """Aggregate metrics across all GPUs using distributed reduction."""
        if not dist.is_initialized():
            # Single GPU case - just compute averages locally
            avg_metrics = {}
            for k, v in local_metrics.items():
                avg_metrics[k] = v / max(batch_count, 1)
            return avg_metrics
        
        world_size = dist.get_world_size()
        
        # Convert metrics to tensors for reduction
        metric_keys = sorted(local_metrics.keys())
        local_sums = torch.tensor(
            [local_metrics[k] for k in metric_keys], 
            dtype=torch.float32, 
            device=self.device
        )
        local_count = torch.tensor(batch_count, dtype=torch.float32, device=self.device)
        
        # Sum metrics across all GPUs
        global_sums = local_sums.clone()
        global_count = local_count.clone()
        dist.all_reduce(global_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        
        # Compute global averages
        avg_metrics = {}
        for i, key in enumerate(metric_keys):
            avg_metrics[key] = global_sums[i] / max(global_count.item(), 1)
        
        return avg_metrics
    
    def _log_metrics(self, metrics, epoch, prefix='Eval'):
        for key in metrics:
            try:
                mlflow.log_metric(f"{prefix}/{key}", metrics[key].item(), epoch)
            except Exception as e:
                logger.error(f"Error while logging to AML: {e}")

    def _log_images(self, imgs, wm_imgs, epoch, prefix='Eval', num_samples=1):
        for i in range(num_samples):
            try:
                img = transforms.ToPILImage()(
                    (imgs[i].cpu() + 1.0)/2.0)
                mlflow.log_image(img, f"{prefix}_inp_epoch_{epoch:04d}_{i:02}.png")
                img = transforms.ToPILImage()(
                    (wm_imgs[i].cpu() + 1.0)/2.0)
                mlflow.log_image(img, f"{prefix}_wm_epoch_{epoch:04d}_{i:02}.png")
                img = 20.0 * (imgs[i].cpu() - wm_imgs[i].cpu())
                img = transforms.ToPILImage()((img + 1.0)/2.0)
                mlflow.log_image(img, f"{prefix}_resx20_epoch_{epoch:04d}_{i:02}.png")
            except Exception as e:
                logger.error(f"Error while logging images to AML: {e}")

    def save_ckpt(self, model_state_dict, epoch, output_dir):
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_state = {
                'model': model_state_dict,
                'epoch': epoch,
            }
            save_path = os.path.join(output_dir, f'ckpt_{epoch}.pth')
            torch.save(save_state, save_path)
            logger.info(f"{save_path} ckpt saved!")
