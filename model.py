import torch
import torch.nn as nn


class Simple_CNN(nn.Module):
    def __init__(self, input, num_classes):
        super(Simple_CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3), # -> 10x222x222
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # -> 10x111x111
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=2), #-> 20x54x54 
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=2), # 40x25x25
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40*25*25, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 224),
            nn.ReLU(),
            nn.Linear(224, 64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.dense(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple_CNN(input=224, num_classes=10)
samples = torch.rand((16, 3,224, 224))
output = model(samples)
print(output.shape)