import torch.nn as nn
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from any_models.backbones import get_backbone

class AnimalClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet50', pretrained=True):
        super(AnimalClassifier, self).__init__()
       
        self.backbone = get_backbone(backbone_name, pretrained)
        
        num_features = self.backbone.get_feature_dim()
        
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x