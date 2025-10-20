import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Input: (3, 64, 64)
    Conv(3->16,k3,s1,p1)+ReLU -> MaxPool(2)
    Conv(16->32,k3,s1,p1)+ReLU -> MaxPool(2)
    Flatten -> FC(8192->100)+ReLU -> FC(100->10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1  = nn.Linear(32*16*16, 100)
        self.fc2  = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,16,32,32)
        x = self.pool(F.relu(self.conv2(x)))  # (B,32,16,16)
        x = x.flatten(1)                      # (B,8192)
        x = F.relu(self.fc1(x))               # (B,100)
        x = self.fc2(x)                       # (B,10)
        return x
