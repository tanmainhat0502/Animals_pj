import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import cv2
from torchvision import transforms
import numpy as np

class Animals_dataset(nn.Module):
    def __init__(self, root, istrain, transform, *args, **kwargs):
        super(Animals_dataset, self).__init__(*args, **kwargs)
        if istrain:
            folder_path = os.path.join(root, "train")
        else: 
            folder_path = os.path.join(root, "test")
        
        self.transform = transform
        self.img = []
        self.label = []
        for idx, class_name in enumerate(os.listdir(folder_path)):
            class_path = os.path.join(folder_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.img.append(img_path)
                self.label.append(idx)        
    def ___len__(self):
        return len(self.label)

    def __getitem__(self, id):
        img = self.img[id]
        label = self.label[id]
        img_read = cv2.imread(img)
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        if self.transform:
            img_read = self.transform(img_read)
        return img_read, label
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])
train_data = Animals_dataset(os.path.join(os.getcwd(), "splited_dataset"), istrain=True, transform=transform)
test_data = Animals_dataset(os.path.join(os.getcwd(), "splited_dataset"), istrain=False, transform=transform)
img, label = train_data[0]
img = img.numpy()
img = np.transpose(img,(1,2,0))
cv2.imshow(f"This is {label} images", img)
cv2.waitKey(0)