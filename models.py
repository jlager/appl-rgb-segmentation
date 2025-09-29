import torch
import segmentation_models_pytorch as smp
import timm
import math

# =============================================================================
# U-Net Segmentation Model
# =============================================================================

class UNet(torch.nn.Module):

    def __init__(
        self, 
        backbone="resnet34",
        num_classes=2,
    ) -> None:
        
        super().__init__()
        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# =============================================================================
# ViT Segmentation Model
# =============================================================================

class ResidualLayer(torch.nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.clone()
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = residual + x
        return x

class ResidualBlock(torch.nn.Module):

    def __init__(self, channels: int, layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ResidualLayer(channels) for _ in range(layers)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
class Upsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.norm = torch.nn.BatchNorm2d(in_channels)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
class ViT(torch.nn.Module):

    def __init__(
        self, 
        backbone='vit_small', # 'vit_small', 'vit_base'
        patch_size=8, # 8, 16
        window_size=224, # 224
        image_size=224, # 224, 448
        pretrained=True, 
        num_classes=2,
        verbose=False,
    ) -> None:
        
        super().__init__()
        self.verbose = verbose

        # build vit encoder
        self.encoder = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(-1,),
            img_size=image_size,
        )

        # get hidden dimension from encoder
        hidden_dim = self.encoder.feature_info.channels()[-1]

        # build segmentation decoder
        self.decoder = self._build_decoder(hidden_dim, patch_size, num_classes)
    
    def _build_decoder(
        self, 
        hidden_dim: int, 
        patch_size: int, 
        num_classes: int,
    ) -> torch.nn.Module:
        
        layers = []
        current_dim = hidden_dim
        
        # compute number of upsampling steps needed
        num_layers = int(math.log2(patch_size))
        
        # build layers
        for i in range(num_layers):
            next_dim = current_dim // 2
            layers.append(Upsample(current_dim, next_dim))
            layers.append(ResidualBlock(next_dim, 2))
            current_dim = next_dim
        
        # final output
        layers.append(torch.nn.BatchNorm2d(current_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(current_dim, num_classes, kernel_size=1))

        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.encoder(x)[0] # [B, C, H, W] -> [B, hidden_dim, H/patch_size, W/patch_size]
        x = self.decoder(x) # [B, hidden_dim, H/patch_size, W/patch_size] -> [B, num_classes, H, W]
        return x
    
# =============================================================================
# Model builder
# =============================================================================

def build_model(
    model_type: str,
    backbone: str,
    tile_size: int,
    device: torch.device,
) -> torch.nn.Module:

    if model_type == 'vit':
        return ViT(
            backbone=backbone,
            patch_size=8 if 'patch8' in backbone else 16,
            window_size=224,
            image_size=tile_size,
            pretrained=True,
            num_classes=2,
        ).to(device)
    
    if model_type == 'unet':
        return UNet(
            backbone=backbone,
            num_classes=2,
        ).to(device)