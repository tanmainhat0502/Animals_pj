import torch.nn as nn
from .resnet import resnet18, resnet34, resnet50
from .inceptionnet import inception_v1
from .efficientnet import efficientnet_b0
from .mobilenet import mobilenet_v2

class Backbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True): 
        super(Backbone, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained)  
            self.feature_dim = 512
        elif self.backbone_name == 'resnet34':
            self.model = resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif self.backbone_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif self.backbone_name == 'inception_v1':
            self.model = inception_v1(pretrained=pretrained)
            self.feature_dim = 1024
        elif self.backbone_name == 'efficientnet_b0':
            self.model = efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif self.backbone_name == 'mobilenet_v2':
            self.model = mobilenet_v2(pretrained=pretrained)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} không được hỗ trợ!")

        # Xóa layer fully connected cuối cùng
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()
    
    def forward(self, x):
        return self.model(x)
    
    def get_feature_dim(self):
        return self.feature_dim

def get_backbone(backbone_name='resnet50', pretrained=True): 
    return Backbone(backbone_name, pretrained)
