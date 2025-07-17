import torch
import timm

class ResidualLayer(torch.nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.norm1 = torch.nn.BatchNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
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

    def __init__(self, channels: int, layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            ResidualLayer(channels) for _ in range(layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Upsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.norm = torch.nn.BatchNorm2d(in_channels)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
class ViTEncoder(torch.nn.Module):
    
    def __init__(self, vit_model='vit_small_patch8_224', pretrained=True, img_size=224):
        super().__init__()
        self.encoder = timm.create_model(
            vit_model, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(-1,),
            img_size=img_size
        )
        
    def forward(self, x):
        return self.encoder(x)[0] # [B, C, H, W] -> [B, hidden_dim, H/8, W/8]

class ViTSegmentation(torch.nn.Module):

    def __init__(self, vit_model='vit_small_patch8_224', pretrained=True, num_classes=2, img_size=224):
        
        super().__init__()

        # vit encoder [B, C, H, W] -> [B, hidden_dim, H/8, W/8]
        self.encoder = ViTEncoder(vit_model, pretrained, img_size)
        
        # get hidden dimension
        if 'base' in vit_model:
            hidden_dim = 768
        elif 'large' in vit_model:
            hidden_dim = 1024
        elif 'small' in vit_model:
            hidden_dim = 384
        elif 'tiny' in vit_model:
            hidden_dim = 192
        else:
            hidden_dim = 768  # Default to base size
        
        # segmentation decoder [B, hidden_dim, H/8, W/8] -> [B, num_classes, H, W]
        self.decoder = torch.nn.Sequential(
            Upsample(hidden_dim, hidden_dim//2),
            ResidualBlock(hidden_dim//2, 2),
            Upsample(hidden_dim//2, hidden_dim//4),
            ResidualBlock(hidden_dim//4, 2),
            Upsample(hidden_dim//4, hidden_dim//8),
            ResidualBlock(hidden_dim//8, 2),
            torch.nn.BatchNorm2d(hidden_dim//8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_dim//8, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.encoder(x) # [B, C, H, W] -> [B, hidden_dim, H/8, W/8]
        x = self.decoder(x) # [B, hidden_dim, H/8, W/8] -> [B, num_classes, H, W]
        return x