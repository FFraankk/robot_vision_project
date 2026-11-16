"""Learned models and hand-written HOG features for Task 7."""
import numpy as np
import torch
from torch import nn


class CharacterCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, classes),
        )

    def forward(self, images):
        return self.layers(images)


class LinearSVM(nn.Module):
    def __init__(self, classes, features=324):
        super().__init__()
        self.linear = nn.Linear(features, classes)

    def forward(self, features):
        return self.linear(features)


def squared_hinge_loss(scores, labels, model, regularization):
    """One-vs-rest linear SVM objective with unregularized intercepts."""
    targets = -torch.ones_like(scores)
    targets.scatter_(1, labels[:, None], 1)
    penalty = torch.relu(1 - targets * scores).square().sum(1).mean()
    return 0.5 * model.linear.weight.square().sum() + regularization * penalty


def hog_features(images, cell_size=8):
    """Unsigned HOG, nine bins and overlapping L2-Hys blocks."""
    if cell_size not in (4, 8):
        raise ValueError("cell_size must be 4 or 8")
    cells_per_side = 32 // cell_size
    features = []
    for image in images:
        dy, dx = np.gradient(image)
        magnitude = np.hypot(dx, dy)
        orientation = (np.arctan2(dy, dx) % np.pi) * 9 / np.pi
        lower = np.floor(orientation).astype(int) % 9
        fraction = orientation - np.floor(orientation)
        histogram = np.zeros((cells_per_side, cells_per_side, 9), dtype=np.float32)
        rows, columns = np.indices(image.shape)
        cells = (rows // cell_size, columns // cell_size)
        np.add.at(histogram, (*cells, lower), magnitude * (1 - fraction))
        np.add.at(histogram, (*cells, (lower + 1) % 9), magnitude * fraction)
        blocks = []
        for row in range(cells_per_side - 1):
            for column in range(cells_per_side - 1):
                block = histogram[row:row + 2, column:column + 2].ravel()
                block = block / np.sqrt(block @ block + 1e-6)
                block = np.minimum(block, 0.2)
                blocks.extend(block / np.sqrt(block @ block + 1e-6))
        features.append(blocks)
    return np.asarray(features, dtype=np.float32)
