"""Image-only heading estimator: outputs (cos theta, sin theta)."""
import torch
from torch import nn


class HeadingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            # Preserve horizontal position: global pooling would discard heading cues.
            nn.Flatten(), nn.Linear(24 * 8 * 16, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.network(images)


def angular_error_degrees(predictions, targets):
    predicted = torch.atan2(predictions[:, 1], predictions[:, 0])
    actual = torch.atan2(targets[:, 1], targets[:, 0])
    difference = predicted - actual
    return torch.rad2deg(torch.atan2(torch.sin(difference), torch.cos(difference)).abs())


def get_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
