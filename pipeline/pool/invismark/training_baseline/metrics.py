
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


def bit_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0.5):
    """Calculate the bit-wise accuracy between true and predicted tensors.

    Args:
        y_true: Ground truth binary tensor.
        y_pred: Predicted tensor with values to be thresholded.
        threshold: Threshold value for binarizing predictions. Defaults to 0.5.

    Returns:
        torch.Tensor: A single-element tensor containing the bit accuracy ratio.
    """
    assert y_true.size() == y_pred.size()
    return torch.Tensor([(y_pred >= threshold).eq(
        y_true >= 0.5).sum().float().item() / y_pred.numel()])


def image_psnr(preds, targets, data_range=2.0):
    """Calculate the Peak Signal-to-Noise Ratio between predicted and target images.

    Args:
        preds: Predicted image tensor.
        targets: Target image tensor.
        data_range: The data range of the input images. Defaults to 2.0.

    Returns:
        torch.Tensor: The PSNR value between the predicted and target images.
    """
    psnr = PeakSignalNoiseRatio(data_range=data_range)
    return psnr(preds, targets)


def image_ssim(preds, targets, data_range=2.0):
    """Calculate the Structural Similarity Index between predicted and target images.

    Args:
        preds: Predicted image tensor.
        targets: Target image tensor.
        data_range: The data range of the input images. Defaults to 2.0.

    Returns:
        torch.Tensor: The SSIM value between the predicted and target images.
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    return ssim(preds, targets)
