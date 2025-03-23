import yaml
from torchvision import transforms

def get_transforms(config_path='configs/transforms.yaml', phase='train'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if phase == 'train':
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]
    
    return transforms.Compose(transform_list)