from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from api_tracer import APITracer


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    output_path = "tools/api_tracer/trace_output_test_module"
    with APITracer(
        dialect="torch", output_path=output_path, levels=[0, 1], merge_output=True
    ) as tracer:
        model = SimpleCNN(num_classes=10)
        sample_input = torch.randn(4, 1, 28, 28)
        output = model(sample_input)
        print("Output shape:", output.shape)
        print("Output probabilities:\n", output)
