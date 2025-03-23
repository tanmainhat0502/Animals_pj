import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Block cho EfficientNet."""
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)
        
        # Expansion phase
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                     padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)
        self.drop_connect_rate = 0.2
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        
        # Squeeze-and-Excitation
        se = self.se(out)
        out = out * se
        
        # Residual connection with drop connect
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                out = self.drop_connect(out, self.drop_connect_rate)
            out += identity
        return out
    
    def drop_connect(self, x, drop_rate):
        keep_prob = 1.0 - drop_rate
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep_prob
        return x / keep_prob * mask.float()

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # Cấu hình EfficientNet-B0
        settings = [
            # (expand_ratio, out_channels, repeats, stride, kernel_size)
            (1, 16, 1, 1, 3),   # Stage 1
            (6, 24, 2, 2, 3),   # Stage 2
            (6, 40, 2, 2, 5),   # Stage 3
            (6, 80, 3, 2, 3),   # Stage 4
            (6, 112, 3, 1, 5),  # Stage 5
            (6, 192, 4, 2, 5),  # Stage 6
            (6, 320, 1, 1, 3),  # Stage 7
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        
        # Build stages
        self.blocks = nn.ModuleList([])
        in_channels = 32
        for expand_ratio, out_channels, repeats, stride, kernel_size in settings:
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(in_channels, out_channels, expand_ratio, kernel_size, block_stride)
                )
                in_channels = out_channels
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def efficientnet_b0():
    """EfficientNet-B0: Baseline của EfficientNet."""
    return EfficientNet()