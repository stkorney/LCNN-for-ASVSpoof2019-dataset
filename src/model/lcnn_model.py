import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        halves = torch.split(x, x.size(1) // 2, dim=1)
        return torch.max(halves[0], halves[1])


class LCNNModel(nn.Module):
    """
    Light Convolutional Neural Network model for spoofing detection.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            MFM(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1, bias=False),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0, bias=False),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1, bias=False),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(32 * 54 * 37, 160),
            MFM(),
            nn.BatchNorm1d(80),
            nn.Dropout(p=0.5),
            nn.Linear(80, 2)
        )

    def forward(self, data_object, **batch):
        return {"logits": self.net(data_object)}


