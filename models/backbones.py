# models/backbone.py
import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50
from .inceptionnet import inception_v1
from .vgg import vgg16
from .alexnet import alexnet
from .efficientnet import efficientnet_b0
from .mobilenet import mobilenet_v2

class Backbone(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(Backbone, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == 'resnet18':
            self.model = resnet18()
            self.feature_dim = 512
        elif self.backbone_name == 'resnet34':
            self.model = resnet34()
            self.feature_dim = 512
        elif self.backbone_name == 'resnet50':
            self.model = resnet50()
            self.feature_dim = 2048
        elif self.backbone_name == 'inception_v1':
            self.model = inception_v1()
            self.feature_dim = 1024
        elif self.backbone_name == 'vgg16':
            self.model = vgg16()
            self.feature_dim = 512 * 7 * 7
        elif self.backbone_name == 'alexnet':
            self.model = alexnet()
            self.feature_dim = 256 * 6 * 6
        elif self.backbone_name == 'efficientnet_b0':
            self.model = efficientnet_b0()
            self.feature_dim = 1280
        elif self.backbone_name == 'mobilenet_v2':
            self.model = mobilenet_v2()
            self.feature_dim = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} không được hỗ trợ!")
        
        # Loại bỏ fully connected layer cuối cùng để chỉ lấy features
        if 'resnet' in self.backbone_name or 'inception' in self.backbone_name or \
           'efficientnet' in self.backbone_name or 'mobilenet' in self.backbone_name:
            self.model.fc = nn.Identity()
        elif self.backbone_name == 'vgg16':
            self.model.classifier = nn.Identity()
        elif self.backbone_name == 'alexnet':
            self.model.classifier = nn.Identity()
    
    def forward(self, x):
        return self.model(x)
    
    def get_feature_dim(self):
        return self.feature_dim

def get_backbone(backbone_name='resnet50'):
    return Backbone(backbone_name)