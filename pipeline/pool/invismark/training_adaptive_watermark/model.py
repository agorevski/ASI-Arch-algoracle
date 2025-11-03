import logging
import torch
from torch import nn, Tensor
import torchvision
from torch.nn import functional as thf
import torchvision.transforms as transforms
import bchlib

import utils
import noise
from adaptive_watermark import AdaptiveWatermark2Image

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class InvisMark(nn.Module):
    def __init__(self, cfg):
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
        images = images.view(-1, 3, *images.shape[-2:])
        return self.decoder(self.img_preprocess(images))

    def forward(self, inputs, secret):
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
    def forward(self, x: Tensor) -> Tensor:
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
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)

class Watermark2Image(nn.Module):
    def __init__(self, watermark_len, resolution=64, hidden_dim=16):
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
        return self.transform(x)


class Conv2d(nn.Module):
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
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels='default',
            out_channels='default',
            activ='relu',
            norm=None):
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
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.watermark2image = AdaptiveWatermark2Image(
            num_bits=cfg['WATERMARK']['NUM_BITS'],
            resolution=cfg['IMAGE']['SIZE'][0],
            d_model=cfg['ENCODER']['ADAPTIVE_WATERMARK']['D_MODEL'])
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
        watermark = self.watermark2image(watermark, image)
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
    def __init__(self, cfg):
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
        return self.main(image)


class Decoder(nn.Module):
    def __init__(self, cfg):
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
        return self.main(image)


class BCHECC:

    def __init__(self, t, m):
        self.t = t  # number of errors to be corrected
        self.m = m  # total of bits n is 2^m
        self.bch = bchlib.BCH(t, m=m)
        self.data_bytes = (self.bch.n + 7) // 8 - self.bch.ecc_bytes
        self.decode_error_count = 0

    def batch_encode(self, batch_size):
        secrets = []
        uuid_bytes = utils.uuid_to_bytes(batch_size)
        for input in uuid_bytes:
            ecc = self.bch.encode(input)
            secrets += [torch.Tensor([int(i)
                                      for i in ''.join(format(x, '08b') for x in input + ecc)])]
            assert len(secrets[-1]) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.vstack(secrets).type(torch.float32)

    def batch_decode_ecc(self, secrets: torch.Tensor, threshold: float = 0.5):
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
        assert len(input) == self.data_bytes, f"Input str length should be {self.data_bytes}"
        input_bytes = bytearray(input, 'utf-8')
        ecc = self.bch.encode(input_bytes)
        packet = input_bytes + ecc
        secret = [int(i) for i in ''.join(format(x, '08b') for x in packet)]
        assert len(secret) == 2**self.m, f"Encoded secret bits length should be {2**self.m}"
        return torch.Tensor(secret).type(torch.float32).unsqueeze(0)

    def decode_str(self, secrets: torch.Tensor, threshold: float = 0.5):
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
        return self.batch_decode_ecc(secrets, threshold)[:, :-self.bch.ecc_bytes * 8]

    def reset_error_count(self):
        """Reset the decode error counter."""
        self.decode_error_count = 0

    def get_error_count(self):
        """Get the current decode error count."""
        return self.decode_error_count

    def _bitstring_to_bytes(self, s):
        return bytearray(int(s, 2).to_bytes(
            (len(s) + 7) // 8, byteorder='big'))
