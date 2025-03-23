# models/model.py
import torch.nn as nn
from backbones import get_backbone

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        super(AnimalClassifier, self).__init__()
        # Lấy backbone
        self.backbone = get_backbone(backbone_name, pretrained)
        # Lấy số features từ backbone
        num_features = self.backbone.get_feature_dim()
        # Fully connected layer cuối cùng
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x