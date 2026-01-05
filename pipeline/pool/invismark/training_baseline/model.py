import logging
import torch
from torch import nn, Tensor
import torchvision
from torch.nn import functional as thf
import torchvision.transforms as transforms
import bchlib

import utils
import noise

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InvisMark(nn.Module):
    """Main watermarking model combining encoder, decoder, and discriminator.

    Args:
        cfg: Configuration dictionary containing model hyperparameters.
    """

    def __init__(self, cfg):
        """Initialize the InvisMark model.

        Args:
            cfg: Configuration dictionary containing IMAGE, ENCODER, DECODER,
                and DISCRIMINATOR settings.
        """
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.discriminator = Discriminator(cfg)
        self.decoder = Decoder(cfg)
        self.img_preprocess = transforms.Compose([
            transforms.Resize(cfg['IMAGE']['SIZE']),
        ])
        self.noiser = noise.Noiser(num_transforms=1)

    def encode(self, inputs, secret):
        """Encode a secret watermark into input images.

        Args:
            inputs: Input image tensor of shape (batch, channels, height, width).
            secret: Secret watermark tensor to embed.

        Returns:
            Tuple of (output, resized_inputs, resized_outputs) where output is
            the watermarked image clamped to [-1, 1].
        """
        # Convert multiple images to the channel dimension.
        inputs = inputs.view(inputs.shape[0], -1, *inputs.shape[-2:])
        resized_inputs = self.img_preprocess(inputs)
        resized_outputs = self.encoder(resized_inputs, secret)
        residual = resized_outputs - resized_inputs
        residual = transforms.Resize(
            inputs.shape[-2:])(residual)
        output = torch.clamp(inputs + residual, min=-1.0, max=1.0)
        return output, resized_inputs, resized_outputs

    def decode(self, images):
        """Decode the watermark from images.

        Args:
            images: Image tensor containing embedded watermark.

        Returns:
            Decoded watermark tensor.
        """
        images = images.view(-1, 3, *images.shape[-2:])
        return self.decoder(self.img_preprocess(images))

    def forward(self, inputs, secret):
        """Forward pass encoding and decoding the watermark.

        Args:
            inputs: Input image tensor.
            secret: Secret watermark tensor to embed.

        Returns:
            Dictionary containing final_outputs, resized_inputs, resized_outputs,
            decode_wm (decoded watermark), and decode_wm_noise (decoded after noise).
        """
        output, resized_inputs, resized_output = self.encode(inputs, secret)
        decode_wm = self.decode(output)
        noised_output = self.noiser(output)
        decode_wm_noise = self.decode(noised_output)
        return {"final_outputs": output.view(-1, 3, *output.shape[-2:]),
                "resized_inputs": resized_inputs.view(-1, 3, *resized_inputs.shape[-2:]),
                "resized_outputs": resized_output.view(-1, 3, *resized_output.shape[-2:]),
                "decode_wm": decode_wm,
                "decode_wm_noise": decode_wm_noise}

class LayerNorm2d(nn.LayerNorm):
    """Layer normalization for 2D inputs (NCHW format)."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization to 2D input.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Normalized tensor of shape (N, C, H, W).
        """
        x = x.permute(0, 2, 3, 1)
        x = thf.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ImageViewLayer(nn.Module):
    """Layer to reshape flattened tensor into image format."""

    def __init__(self, hidden_dim=16, channel=3):
        """Initialize ImageViewLayer.

        Args:
            hidden_dim: Spatial dimension of output image. Defaults to 16.
            channel: Number of output channels. Defaults to 3.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        """Reshape input tensor to image format.

        Args:
            x: Flattened input tensor.

        Returns:
            Reshaped tensor of shape (-1, channel, hidden_dim, hidden_dim).
        """
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)

class Watermark2Image(nn.Module):
    """Transform watermark vector into image-like tensor."""

    def __init__(self, watermark_len, resolution=64, hidden_dim=16):
        """Initialize Watermark2Image module.

        Args:
            watermark_len: Length of input watermark vector.
            resolution: Target output resolution. Defaults to 64.
            hidden_dim: Hidden spatial dimension before upsampling. Defaults to 16.

        Raises:
            AssertionError: If resolution is not divisible by hidden_dim.
        """
        super().__init__()
        assert resolution % hidden_dim == 0, "Resolution should be divisible by hidden_dim"
        self.transform = nn.Sequential(
            nn.Linear(watermark_len, hidden_dim * hidden_dim * 3),
            ImageViewLayer(hidden_dim, channel=3),
            nn.Upsample(
                scale_factor=(
                    resolution // hidden_dim,
                    resolution // hidden_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        """Transform watermark to image representation.

        Args:
            x: Watermark tensor of shape (batch, watermark_len).

        Returns:
            Image tensor of shape (batch, 3, resolution, resolution).
        """
        return self.transform(x)


class Conv2d(nn.Module):
    """Convolution block with optional activation and normalization."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            activ='relu',
            norm=None):
        """Initialize Conv2d block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of convolution kernel.
            stride: Convolution stride. Defaults to 1.
            padding: Convolution padding. Defaults to 0.
            bias: Whether to use bias. Defaults to True.
            activ: Activation function ('relu', 'silu', 'tanh', 'leaky_relu', or None).
                Defaults to 'relu'.
            norm: Normalization type ('bn' for BatchNorm or None). Defaults to None.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias)
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'silu':
            self.activ = nn.SiLU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'leaky_relu':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None

        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Apply convolution, normalization, and activation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after conv, norm, and activation.
        """
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    """Decoder block with upsampling and skip connections."""

    def __init__(
            self,
            in_channels,
            skip_channels='default',
            out_channels='default',
            activ='relu',
            norm=None):
        """Initialize decoder block.

        Args:
            in_channels: Number of input channels.
            skip_channels: Number of skip connection channels. Defaults to in_channels // 2.
            out_channels: Number of output channels. Defaults to in_channels // 2.
            activ: Activation function type. Defaults to 'relu'.
            norm: Normalization type. Defaults to None.
        """
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels // 2
        if out_channels == 'default':
            out_channels = in_channels // 2
        self.up = nn.Upsample(scale_factor=(2, 2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2,
                            1, 0, activ=activ, norm=norm)
        self.conv2 = Conv2d(
            out_channels + skip_channels,
            out_channels, 3, 1, 1,
            activ=activ,
            norm=norm)

    def forward(self, x, skip):
        """Forward pass with upsampling and skip connection.

        Args:
            x: Input tensor from previous layer.
            skip: Skip connection tensor from encoder.

        Returns:
            Output tensor after upsampling and concatenation with skip.
        """
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    """U-Net style encoder for embedding watermarks into images."""

    def __init__(self, cfg):
        """Initialize the encoder.

        Args:
            cfg: Configuration dictionary with WATERMARK, IMAGE, and ENCODER settings.
        """
        super().__init__()
        self.cfg = cfg
        self.watermark2image = Watermark2Image(
            cfg['WATERMARK']['NUM_BITS'],
            cfg['IMAGE']['SIZE'][0],
            cfg['ENCODER']['HIDDEN_DIM'])
        # input_channel: 3 from image + 3 from watermark
        self.pre = Conv2d(3 + 3 * cfg['ENCODER']['NUM_FRAMES'],
                          cfg['ENCODER']['NUM_INITIAL_CHANNELS'], 3, 1, 1)
        self.enc = nn.ModuleList()
        input_channel = cfg['ENCODER']['NUM_INITIAL_CHANNELS']
        for _ in range(cfg['ENCODER']['NUM_DOWN_LEVELS']):
            self.enc.append(Conv2d(input_channel, input_channel * 2, 3, 2, 1))
            input_channel *= 2

        self.dec = nn.ModuleList()
        for i in range(cfg['ENCODER']['NUM_DOWN_LEVELS']):
            skip_width = input_channel // 2 if i < cfg['ENCODER']['NUM_DOWN_LEVELS'] - \
                1 else input_channel // 2 + 3 + 3 * cfg['ENCODER']['NUM_FRAMES']  # 3 image channel + 3 watermark channel
            self.dec.append(
                DecBlock(input_channel, skip_width, activ='relu', norm='none'))
            input_channel //= 2

        self.post = nn.Sequential(
            Conv2d(input_channel, input_channel, 3, 1, 1, activ='None'),
            Conv2d(input_channel, input_channel // 2, 1, 1, 0, activ='silu'),
            # Conv2d(input_channel//2, input_channel//4, 1, 1, 0, activ='silu'),
            Conv2d(input_channel // 2,
                   3 * cfg['ENCODER']['NUM_FRAMES'], 1, 1, 0, activ='tanh')
        )

    def forward(self, image: torch.Tensor, watermark):
        """Encode watermark into image.

        Args:
            image: Input image tensor.
            watermark: Watermark tensor to embed.

        Returns:
            Watermarked image tensor.
        """
        watermark = self.watermark2image(watermark)
        inputs = torch.cat((image, watermark), dim=1)

        enc = []
        x = self.pre(inputs)
        for layer in self.enc:
            enc.append(x)
            x = layer(x)

        enc = enc[::-1]
        for i, (layer, skip) in enumerate(zip(self.dec, enc)):
            if i < self.cfg['ENCODER']['NUM_DOWN_LEVELS'] - 1:
                x = layer(x, skip)
            else:
                x = layer(x, torch.cat([skip, inputs], dim=1))
        return self.post(x)


class Discriminator(nn.Module):
    """ResNet-18 based discriminator for watermark detection."""

    def __init__(self, cfg):
        """Initialize the discriminator.

        Args:
            cfg: Configuration dictionary with DISCRIMINATOR settings.
        """
        super().__init__()
        self.extractor = torchvision.models.resnet18(
            weights=cfg['DISCRIMINATOR']['NAME'])
        self.extractor.fc = nn.Linear(
            self.extractor.fc.in_features,
            cfg['DISCRIMINATOR']['NUM_CLASSES'])
        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        """Classify image as watermarked or not.

        Args:
            image: Input image tensor.

        Returns:
            Sigmoid probability of image being watermarked.
        """
        return self.main(image)


class Decoder(nn.Module):
    """ConvNeXt-based decoder for extracting watermarks from images."""

    def __init__(self, cfg):
        """Initialize the decoder.

        Args:
            cfg: Configuration dictionary with DECODER and WATERMARK settings.
        """
        super().__init__()

        self.extractor = torchvision.models.convnext_base(
            weights=cfg['DECODER']['NAME'])
        n_inputs = None
        for name, child in self.extractor.named_children():
            if name == 'classifier':
                for sub_name, sub_child in child.named_children():
                    if sub_name == '2':
                        n_inputs = sub_child.in_features

        self.extractor.classifier = nn.Sequential(
            LayerNorm2d(n_inputs, eps=1e-6),
            nn.Flatten(1),
            nn.Linear(in_features=n_inputs,
                      out_features=cfg['WATERMARK']['NUM_BITS']),
        )

        self.main = nn.Sequential(
            self.extractor,
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        """Extract watermark from image.

        Args:
            image: Input watermarked image tensor.

        Returns:
            Decoded watermark tensor with sigmoid activation.
        """
        return self.main(image)


class BCHECC:
    """BCH error correction code wrapper for watermark encoding/decoding."""

    def __init__(self, t, m):
        """Initialize BCH error correction codec.

        Args:
            t: Number of errors to be corrected.
            m: Total bits n is 2^m.
        """
        self.t = t  # number of errors to be corrected
        self.m = m  # total of bits n is 2^m
        self.bch = bchlib.BCH(t, m=m)
        self.data_bytes = (self.bch.n + 7) // 8 - self.bch.ecc_bytes
        self.decode_error_count = 0

    def batch_encode(self, batch_size):
        """Encode a batch of random UUIDs with BCH error correction.

        Args:
            batch_size: Number of secrets to generate.

        Returns:
            Tensor of shape (batch_size, 2^m) containing encoded secrets.
        """
        secrets = []
        uuid_bytes = utils.uuid_to_bytes(batch_size)
        for input in uuid_bytes:
            ecc = self.bch.encode(input)
            secrets += [torch.Tensor([int(i)
                                      for i in ''.join(format(x, '08b') for x in input + ecc)])]
            assert len(secrets[-1]) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.vstack(secrets).type(torch.float32)

    def batch_decode_ecc(self, secrets: torch.Tensor, threshold: float = 0.5):
        """Decode batch of secrets with BCH error correction.

        Args:
            secrets: Tensor of encoded secrets.
            threshold: Threshold for binarizing soft bits. Defaults to 0.5.

        Returns:
            Tensor of corrected secret bits.
        """
        res = []
        for i in range(len(secrets)):
            packet = self._bch_correct(secrets[i], threshold)
            data_bits = [
                int(k) for k in ''.join(
                    format(
                        x, '08b') for x in packet)]
            res.append(torch.Tensor(data_bits).type(torch.float32))
        return torch.vstack(res)

    def encode_str(self, input: str):
        """Encode a string with BCH error correction.

        Args:
            input: String to encode, must be exactly data_bytes long.

        Returns:
            Tensor of shape (1, 2^m) containing encoded secret bits.

        Raises:
            AssertionError: If input length doesn't match data_bytes.
        """
        assert len(input) == self.data_bytes, f"Input str length should be {self.data_bytes}"
        input_bytes = bytearray(input, 'utf-8')
        ecc = self.bch.encode(input_bytes)
        packet = input_bytes + ecc
        secret = [int(i) for i in ''.join(format(x, '08b') for x in packet)]
        assert len(secret) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.Tensor(secret).type(torch.float32).unsqueeze(0)

    def decode_str(self, secrets: torch.Tensor, threshold: float = 0.5):
        """Decode secrets back to strings with error correction.

        Args:
            secrets: Tensor of encoded secrets.
            threshold: Threshold for binarizing soft bits. Defaults to 0.5.

        Returns:
            Tuple of (n_errs, res) where n_errs is list of error counts (-1 if
            uncorrectable) and res is list of decoded strings (empty if failed).
        """
        n_errs, res = [], []
        for i in range(len(secrets)):
            bit_string = ''.join(str(int(k >= threshold)) for k in secrets[i])
            packet = self._bitstring_to_bytes(bit_string)
            data, ecc = packet[:-self.bch.ecc_bytes],
            packet[-self.bch.ecc_bytes:]
            n_err = self.bch.decode(data, ecc)
            if n_err < 0:
                n_errs.append(n_err)
                res.append([])
                continue
            self.bch.correct(data, ecc)
            packet = data + ecc
            try:
                n_errs.append(n_err)
                res.append(packet[:-self.bch.ecc_bytes].decode('utf-8'))
            except BaseException:
                n_errs.append(-1)
                res.append([])
        return n_errs, res

    def _bch_correct(self, secret: torch.Tensor, threshold: float = 0.5):
        """Apply BCH error correction to a single secret.

        Args:
            secret: Single secret tensor.
            threshold: Threshold for binarizing soft bits. Defaults to 0.5.

        Returns:
            Corrected packet as bytes.
        """
        bitstring = ''.join(str(int(x >= threshold)) for x in secret)
        packet = self._bitstring_to_bytes(bitstring)
        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        n_err = self.bch.decode(data, ecc)
        if n_err < 0:
            self.decode_error_count += 1
            # logger.info("n_err < 0. Cannot accurately decode the message.")
            return packet
        self.bch.correct(data, ecc)
        return bytes(data + ecc)

    def _decode_data_bits(self, secrets: torch.Tensor, threshold: float = 0.5):
        """Decode secrets and return only data bits (excluding ECC bits).

        Args:
            secrets: Tensor of encoded secrets.
            threshold: Threshold for binarizing soft bits. Defaults to 0.5.

        Returns:
            Tensor of data bits without ECC portion.
        """
        return self.batch_decode_ecc(secrets, threshold)[:, :-self.bch.ecc_bytes * 8]

    def reset_error_count(self):
        """Reset the decode error counter."""
        self.decode_error_count = 0

    def get_error_count(self):
        """Get the current decode error count."""
        return self.decode_error_count

    def _bitstring_to_bytes(self, s):
        """Convert binary string to bytearray.

        Args:
            s: Binary string (e.g., '10110101').

        Returns:
            Bytearray representation of the binary string.
        """
        return bytearray(int(s, 2).to_bytes(
            (len(s) + 7) // 8, byteorder='big'))
