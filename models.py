import timm
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import math
from sam3 import build_sam3_image_model

"""
Rework for models.py that adds support for SAM/SAM2 models hosted by timm
Available models:

sam2_hiera_base_plus
sam2_hiera_large
sam2_hiera_small
sam2_hiera_tiny
samvit_base_patch16
samvit_huge_patch16
samvit_large_patch16


The following changes have been made so far:

1. class Vit() --> class SegmentationModel()
    
    - Removed argument for "window_size". Dead parameter
    - Removed argument for patch_size. Some models don't specify this in their model name.
      It can be analyzed from the architecture but I decided to compute it from the model itself.
      See new helper function "_get_patch_size()"
    - Added a list of supported backbones to assert the class argument "backbone"
    - Added separate class calls based on model_type. This is because some attributes aren't shared across all model types.

TODO:

1. SAM3
    - Support for this model is a work in progress
    - Learn how timm is able to interpolate the positional embedding grid. See if I can implement for SAM3
      This would allow us to train on various image sizes
    - Designed with a patch size of 14x14. This results in a decoder output of size 576x576. The final result is interpolated to the original
    

NOTE:

1. SAM
    - samvit pre-trained model checkpoints expect specific image sizes to be depending on model
        * samvit_base_patch16: 1024x1024
        * samvit_large_patch16: 1024x1024
        * samvit_huge_patch16: 1024x1024
    - Despite the above, we can train on various image sizes given that we create the model w/ image_size=1024
    - samvit_base_patch16_224 does not come with pre-trained weights

2. SAM2 
    - T, S, & B+ are configured around image sizes 896x896 while L is configured around 1024x1024
    - These models can also be tuned with different tile sizes

3. SAM3

    - Hardcoded BPE path based on a clone in another dir. Doesn't seem to be installed when pip installing sam3 which is a bug
    - This model can only be tuned on 1008x1008 images/tiles afaik
    - Encoder entrypoint is @ model.backbone.forward_image. The output isn't in the tensor shape we expect for the decoder
      but it can be reshaped to fit.
    - Checkpoint path currently hardcoded
"""

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
    
# =============================================================================
# Segmentation Model
# =============================================================================

class SegmentationModel(torch.nn.Module):
    def __init__(
        self,
        model_type=None,
        backbone=None,   
        checkpoint_path=None,
        image_size=224,        
        pretrained=True, 
        num_classes=2,
        verbose=False,
    ) -> None:
        
        super().__init__()
        extra_kwargs = {}
        if model_type == 'vit':
            extra_kwargs['image_size'] = image_size
        
        elif model_type == 'sam':
            extra_kwargs['img_size'] = 1024

        # Build segmentation encoder
        self.encoder = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(-1,),
            **extra_kwargs,
        )

        # When working w/ ViT backbones, we'll want to get patch size. _get_patch_size() obtains this but needs image dimensions.
        # Grab dimensions from extra_kwargs
        if model_type != 'unet':
            self.image_size = extra_kwargs.get('image_size', extra_kwargs.get('img_size', image_size))


        self.verbose = verbose


        hidden_dim = self.encoder.feature_info.channels()[-1]
        patch_size = self._get_patch_size()
        self.decoder = self._build_decoder(hidden_dim, patch_size, num_classes)

        # # build segmentation encoder (Pytorch Image Models ViT)
        # if model_type == 'vit':
        #     self.encoder = timm.create_model(
        #         backbone, 
        #         pretrained=pretrained, 
        #         features_only=True,
        #         out_indices=(-1,),
        #         img_size=image_size,
        #     )

        # # build segmentation encoder (Pytorch Image Models SAM)
        # elif model_type == 'sam':
        #     self.encoder = timm.create_model(
        #         backbone, 
        #         pretrained=pretrained, 
        #         features_only=True,
        #         out_indices=(-1,),
        #         img_size=1024,
        #     )
        
        # # build segmentation encoder (Pytorch Image Models SAM2)
        # elif model_type == 'sam2':
        #     self.encoder = timm.create_model(
        #         backbone, 
        #         pretrained=pretrained, 
        #         features_only=True,
        #         out_indices=(-1,),
        #     )
        
        # # build segmentation encoder (Meta SAM3)
        # # bpe_path is hardcoded b/c from a cloned directory b/c my environment install doesn't have it
        # elif model_type == 'sam3':
        #     self.model = build_sam3_image_model(
        #         bpe_path="/mnt/DGX01/Personal/milliganj/codebase/gits/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        #         eval_mode=False,
        #         device=torch.device("cpu"),  # Moved to GPU later.
        #         checkpoint_path=checkpoint_path,
        #         load_from_HF=False,
        #         enable_segmentation=True
        #     )
        # else:
        #     raise ValueError(f"Model type {model_type} not supported")


        # 1. get hidden dimension from encoder
        # 2. determine initial downsampling factor (patch size) from output stride
        # # 3. build segmentation decoder
        # if model_type == 'sam3':
        #     self.encoder = self.model.backbone.forward_image    # Entrypoint for the SAM3 encoder
        #     hidden_dim = self.model.hidden_dim
        # else:
        #     hidden_dim = self.encoder.feature_info.channels()[-1]
    

        if verbose:
            print(f"Model initialized with {backbone} as encoder and {num_classes} classes")
            print(f"    - Patch size: {patch_size}x{patch_size}")
            print(f"    - Hidden dimension: {hidden_dim}")

    def _get_patch_size(self): 
        # patch size is equivalent to the initial downsampling factor a.k.a output stride
        # output stride can be thought of as the height of the initial input // height of the resulting feature map
        # because we take out_indices=(-1,) we'll just have the result from the final feature map
        H, W = self.image_size, self.image_size
        dummy = torch.randn(1, 3, H, W)
        outputs = self.encoder(dummy)
        # if self.model_type == 'sam3':
        #     outputs = outputs["backbone_fpn"][-1]
        H_hat, W_hat = outputs[0].shape[-2], outputs[0].shape[-1]
        output_stride = H // H_hat
        assert output_stride == W // W_hat, "Output stride mismatch" # make more descriptive
        return output_stride

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
        for _ in range(num_layers):
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

        # # If using SAM3, the encoder output isn't a set of feature maps like the timms models toss out
        # # We'll need to extract an actual feature map to sent to the decoder
        # if self.model_type == 'sam3':
        #     x = self.encoder(x)
        #     x = x["backbone_fpn"][-1]  # [B, C, H, W] -> [B, hidden_dim, H/patch_size, W/patch_size]
        #     x = self.decoder(x)
        #     x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        # else:

        x = self.encoder(x)[-1]    # [B, C, H, W] -> [B, hidden_dim, H/patch_size, W/patch_size]
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
    verbose: bool = True,
) -> torch.nn.Module:

    model_types = ["vit", "sam", "sam2", "sam3", "unet"]
    assert model_type in model_types, \
        f"Model type {model_type} not found in list of supported model types. Select from:\n \
        {model_types}"

    if model_type == 'vit':
        return SegmentationModel(
            model_type="vit",
            backbone=backbone,
            image_size=tile_size,
            pretrained=True,
            num_classes=2,
            verbose=verbose,
        ).to(device)

    elif model_type == 'sam':
        return SegmentationModel(
            model_type="sam",
            backbone=backbone,
            image_size=tile_size,
            pretrained=True,
            num_classes=2,
            verbose=verbose,
        ).to(device)

    elif model_type == 'sam2':
        return SegmentationModel(
            model_type="sam2",
            backbone=backbone,
            pretrained=True,
            num_classes=2,
            verbose=verbose,
        ).to(device)

    elif model_type == 'sam3':
        return SegmentationModel(
            model_type="sam3",
            checkpoint_path="/mnt/DGX01/Personal/milliganj/codebase/gits/sam3/sam3/sam3.pt",
            image_size=tile_size,
            verbose=verbose,
        ).to(device)

    elif model_type == 'unet':
        return UNet(
            backbone=backbone,
            num_classes=2,
        ).to(device)
        
    else:
        raise ValueError(f"Model type {model_type} not supported")