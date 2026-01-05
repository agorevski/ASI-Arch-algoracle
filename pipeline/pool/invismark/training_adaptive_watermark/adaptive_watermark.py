import torch.nn as nn
import torch.nn.functional as F

class ImageViewLayer(nn.Module):
    """Reshape linear output to image format."""

    def __init__(self, hidden_dim=16, channel=3):
        """Initialize the ImageViewLayer.

        Args:
            hidden_dim: Spatial dimension of the output image. Defaults to 16.
            channel: Number of output channels. Defaults to 3.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        """Reshape flat tensor to image format.

        Args:
            x: Input tensor of shape [batch, hidden_dim * hidden_dim * channel].

        Returns:
            Reshaped tensor of shape [batch, channel, hidden_dim, hidden_dim].
        """
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)


class Watermark2Image(nn.Module):
    """Original watermark generation - preserves bit information."""

    def __init__(self, watermark_len, resolution=64, hidden_dim=16):
        """Initialize the Watermark2Image module.

        Args:
            watermark_len: Length of the input watermark bit vector.
            resolution: Output image resolution. Defaults to 64.
            hidden_dim: Intermediate spatial dimension before upsampling.
                Defaults to 16.

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
        """Transform watermark bits into a spatial watermark image.

        Args:
            x: Input watermark tensor of shape [batch, watermark_len].

        Returns:
            Watermark image tensor of shape [batch, 3, resolution, resolution].
        """
        return self.transform(x)


class LightweightImageEncoder(nn.Module):
    """Lightweight CNN to extract image features for attention map generation."""

    def __init__(self, in_channels=3):
        """Initialize the LightweightImageEncoder.

        Args:
            in_channels: Number of input image channels. Defaults to 3.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            # Initial conv
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Downsample block 1
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample block 2
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        """Extract feature maps from input image.

        Args:
            image: Input image tensor of shape [batch, 3, H, W].

        Returns:
            Feature tensor of shape [batch, 128, H/4, W/4].
        """
        return self.encoder(image)


class ContentAttentionHead(nn.Module):
    """Generate spatial attention map based on image content."""

    def __init__(self, in_channels=128):
        """Initialize the ContentAttentionHead.

        Args:
            in_channels: Number of input feature channels. Defaults to 128.
        """
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, features):
        """Generate spatial attention map from feature tensor.

        Args:
            features: Feature tensor of shape [batch, 128, H/4, W/4].

        Returns:
            Attention map of shape [batch, 1, H/4, W/4] with values in [0, 1].
        """
        return self.attention_net(features)


class HybridAdaptiveWatermark2Image(nn.Module):
    """Hybrid content-adaptive watermark embedding.

    Combines:
        1. Original Watermark2Image for reliable bit-to-spatial mapping
        2. Lightweight content-based attention for adaptive embedding strength

    This preserves BitAcc while improving perceptual quality.
    """

    def __init__(self, num_bits=256, resolution=256, hidden_dim=16,
                 min_strength=0.7, max_strength=1.3):
        """Initialize the HybridAdaptiveWatermark2Image module.

        Args:
            num_bits: Length of the watermark bit vector. Defaults to 256.
            resolution: Output watermark resolution. Defaults to 256.
            hidden_dim: Intermediate spatial dimension for base watermark.
                Defaults to 16.
            min_strength: Minimum embedding strength for smooth regions.
                Defaults to 0.7.
            max_strength: Maximum embedding strength for textured regions.
                Defaults to 1.3.
        """
        super().__init__()
        self.num_bits = num_bits
        self.resolution = resolution
        self.min_strength = min_strength
        self.max_strength = max_strength

        # Original watermark generator (preserves bit information)
        self.base_watermark = Watermark2Image(
            watermark_len=num_bits,
            resolution=resolution,
            hidden_dim=hidden_dim
        )

        # Lightweight image feature extractor
        self.image_encoder = LightweightImageEncoder(in_channels=3)

        # Content attention head
        self.attention_head = ContentAttentionHead(in_channels=128)

    def forward(self, watermark_bits, image):
        """Generate content-adaptive watermark from bits and image.

        The watermark embedding strength is modulated based on image content:
            - Smooth regions receive min_strength embedding
            - Textured regions receive max_strength embedding
            - No region has zero embedding to preserve decodability

        Args:
            watermark_bits: Binary watermark tensor of shape [batch, num_bits].
            image: Input image tensor of shape [batch, 3, H, W].

        Returns:
            Content-modulated watermark of shape [batch, 3, H, W].
        """
        # 1. Generate base watermark (preserves all bit information)
        base_watermark = self.base_watermark(watermark_bits)

        # 2. Extract image features
        image_features = self.image_encoder(image)

        # 3. Generate content-based attention map
        attention_map = self.attention_head(image_features)

        # 4. Upsample attention map to match watermark resolution
        attention_map = F.interpolate(
            attention_map,
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=False
        )

        # 5. Map attention from [0, 1] to [min_strength, max_strength]
        strength_range = self.max_strength - self.min_strength
        attention_map = self.min_strength + strength_range * attention_map

        # 6. Apply content-adaptive modulation
        adaptive_watermark = base_watermark * attention_map

        return adaptive_watermark


class AdaptiveWatermark2Image(nn.Module):
    """Alias for HybridAdaptiveWatermark2Image for backward compatibility.

    This is the recommended implementation that balances BitAcc and
    perceptual quality.
    """

    def __init__(self, num_bits=256, resolution=256, d_model=128):
        """Initialize the AdaptiveWatermark2Image module.

        Args:
            num_bits: Length of the watermark bit vector. Defaults to 256.
            resolution: Output watermark resolution. Defaults to 256.
            d_model: Unused parameter kept for backward compatibility.
                Defaults to 128.
        """
        super().__init__()
        self.impl = HybridAdaptiveWatermark2Image(
            num_bits=num_bits,
            resolution=resolution,
            hidden_dim=16,
            min_strength=0.7,
            max_strength=1.3
        )

    def forward(self, watermark_bits, image):
        """Generate content-adaptive watermark.

        Args:
            watermark_bits: Binary watermark tensor of shape [batch, num_bits].
            image: Input image tensor of shape [batch, 3, H, W].

        Returns:
            Content-modulated watermark of shape [batch, 3, H, W].
        """
        return self.impl(watermark_bits, image)
