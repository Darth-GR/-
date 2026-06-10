import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import numpy as np
np.random.seed(0)


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        super().__init__()
        self.conv2d = nn.Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            device=device,
            dtype=dtype,
        )
        self.bn = nn.BatchNorm2d(
            dim=out_channels,
            device=device,
            dtype=dtype,
        )
        self.relu = nn.ReLU()

    def forward(self, x: ndl.Tensor):
        return self.relu(self.bn(self.conv2d(x)))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10, num_blocks=2, device=None, dtype="float32"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device, dtype=dtype),
            nn.ReLU(),
            *[
                nn.Residual(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                        nn.BatchNorm1d(hidden_dim, device=device, dtype=dtype),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype),
                    )
                )
                for _ in range(num_blocks)
            ],
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes, device=device, dtype=dtype),
        )

    def forward(self, x):
        if len(x.shape) > 2:
            x = nn.Flatten()(x)
        return self.net(x)


class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.net = nn.Sequential(
            ConvBatchNorm(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBatchNorm(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(nn.Sequential(
                ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
                ConvBatchNorm(32, 32, 3, 1, device=device, dtype=dtype),
            )),
            ConvBatchNorm(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBatchNorm(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(nn.Sequential(
                ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
                ConvBatchNorm(128, 128, 3, 1, device=device, dtype=dtype),
            )),
            nn.Flatten(),
            nn.Linear(128, 128, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.net(x)
