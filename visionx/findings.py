"""Educational binary chest X-ray classifier trained on PneumoniaMNIST."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


class SmallXRayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.25), nn.Linear(96, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def prepare_xray(image: Image.Image) -> torch.Tensor:
    """Convert an image to the same single-channel 224px input used for training."""
    image = image.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def load_findings_model(path: str | Path = "models/pneumonia_mnist.pt"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallXRayCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict_pneumonia(model: nn.Module, image: Image.Image) -> float:
    return float(torch.sigmoid(model(prepare_xray(image))).item())
