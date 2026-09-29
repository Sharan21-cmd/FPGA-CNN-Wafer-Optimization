import torch
import torch.nn as nn


class WaferCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 1 x 48 x 48
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 x 24 x 24
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 x 12 x 12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64 x 6 x 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = WaferCNN()

    x = torch.randn(1, 1, 48, 48)
    y = model(x)

    print("===== MODEL CHECK =====")
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Parameters  :", sum(p.numel() for p in model.parameters()))
