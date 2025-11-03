import torch.nn as nn
import torch.nn.functional as F

class ImageViewLayer(nn.Module):
    """Reshape linear output to image format."""
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)


class Watermark2Image(nn.Module):
    """Original watermark generation - preserves bit information."""
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


class LightweightImageEncoder(nn.Module):
    """Lightweight CNN to extract image features for attention map generation."""
    def __init__(self, in_channels=3):
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
        """
        Args:
            image: [batch, 3, H, W]
        Returns:
            features: [batch, 128, H/4, W/4]
        """
        return self.encoder(image)


class ContentAttentionHead(nn.Module):
    """Generate spatial attention map based on image content."""
    def __init__(self, in_channels=128):
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
        """
        Args:
            features: [batch, 128, H/4, W/4]
        Returns:
            attention_map: [batch, 1, H/4, W/4] in range [0, 1]
        """
        return self.attention_net(features)


class HybridAdaptiveWatermark2Image(nn.Module):
    """
    Hybrid content-adaptive watermark embedding.
    
    Combines:
    1. Original Watermark2Image for reliable bit-to-spatial mapping
    2. Lightweight content-based attention for adaptive embedding strength
    
    This preserves BitAcc while improving perceptual quality.
    """
    def __init__(self, num_bits=256, resolution=256, hidden_dim=16, 
                 min_strength=0.7, max_strength=1.3):
        super().__init__()
        self.num_bits = num_bits
        self.resolution = resolution
        self.min_strength = min_strength  # Minimum embedding strength
        self.max_strength = max_strength  # Maximum embedding strength
        
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
        """
        Args:
            watermark_bits: [batch, num_bits] - binary watermark
            image: [batch, 3, H, W] - input image
            
        Returns:
            adaptive_watermark: [batch, 3, H, W] - content-modulated watermark
        """
        # 1. Generate base watermark (preserves all bit information)
        base_watermark = self.base_watermark(watermark_bits)  # [batch, 3, resolution, resolution]
        
        # 2. Extract image features
        image_features = self.image_encoder(image)  # [batch, 128, H/4, W/4]
        
        # 3. Generate content-based attention map
        attention_map = self.attention_head(image_features)  # [batch, 1, H/4, W/4]
        
        # 4. Upsample attention map to match watermark resolution
        attention_map = F.interpolate(
            attention_map,
            size=(self.resolution, self.resolution),
            mode='bilinear',
            align_corners=False
        )  # [batch, 1, resolution, resolution]
        
        # 5. Map attention from [0, 1] to [min_strength, max_strength]
        # This ensures:
        # - Smooth regions: embedding strength = min_strength (e.g., 0.7x)
        # - Textured regions: embedding strength = max_strength (e.g., 1.3x)
        # - No region has zero embedding (preserves decodability)
        strength_range = self.max_strength - self.min_strength
        attention_map = self.min_strength + strength_range * attention_map
        
        # 6. Apply content-adaptive modulation
        adaptive_watermark = base_watermark * attention_map
        
        return adaptive_watermark


class AdaptiveWatermark2Image(nn.Module):
    """
    Alias for HybridAdaptiveWatermark2Image for backward compatibility.
    This is the recommended implementation that balances BitAcc and perceptual quality.
    """
    def __init__(self, num_bits=256, resolution=256, d_model=128):
        super().__init__()
        # Use hybrid implementation with sensible defaults
        # d_model parameter is ignored for compatibility but not used
        self.impl = HybridAdaptiveWatermark2Image(
            num_bits=num_bits,
            resolution=resolution,
            hidden_dim=16,
            min_strength=0.7,
            max_strength=1.3
        )
        
    def forward(self, watermark_bits, image):
        return self.impl(watermark_bits, image)
