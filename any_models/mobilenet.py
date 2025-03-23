# models/mobilenet.py
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    """Inverted Residual Block cho MobileNetV2."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.use_residual:
            out += identity
        return out

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        
        # Cấu hình MobileNetV2
        settings = [
            # (expand_ratio, out_channels, repeats, stride)
            (1, 16, 1, 1),   # Stage 1
            (6, 24, 2, 2),   # Stage 2
            (6, 32, 3, 2),   # Stage 3
            (6, 64, 4, 2),   # Stage 4
            (6, 96, 3, 1),   # Stage 5
            (6, 160, 3, 2),  # Stage 6
            (6, 320, 1, 1),  # Stage 7
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        
        # Build stages
        self.blocks = nn.ModuleList([])
        in_channels = 32
        for expand_ratio, out_channels, repeats, stride in settings:
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    InvertedResidual(in_channels, out_channels, block_stride, expand_ratio)
                )
                in_channels = out_channels
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
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

def mobilenet_v2():
    """MobileNetV2: Lightweight CNN với inverted residuals."""
    return MobileNetV2()