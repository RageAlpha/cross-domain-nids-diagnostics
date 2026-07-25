"""PyTorch modules for the H3 DANN representation probe.

The architecture follows the H3 protocol:
  - shared MLP feature extractor: input -> 256 -> 128 -> 64
  - binary label predictor: 64 -> 2
  - domain classifier: GRL -> 64 -> 32 -> 2
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer with a fixed alpha.

    The H3 protocol uses GRL(alpha=1.0) and applies lambda only to the
    domain loss. Do not pass lambda as alpha here.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BinaryLabelPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Linear(64, 2)

    def forward(self, z):
        return self.network(z)


class DomainClassifier(nn.Module):
    def __init__(self, grl_alpha=1.0, dropout=0.0):
        super().__init__()
        layers = [
            GradientReversal(alpha=grl_alpha),
            nn.Linear(64, 64),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        ])
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class BinaryMLP(nn.Module):
    """Source-only binary MLP baseline with the same feature extractor."""

    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.feature_extractor = MLPFeatureExtractor(input_dim, dropout=dropout)
        self.label_predictor = BinaryLabelPredictor()

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits = self.label_predictor(features)
        if return_features:
            return logits, features
        return logits


class DANN(nn.Module):
    """Binary MLP-DANN with GRL(alpha=1.0).

    The training loop should compute:
        total_loss = label_loss + lambda_value * domain_loss
    """

    def __init__(self, input_dim, dropout=0.0, grl_alpha=1.0):
        super().__init__()
        self.feature_extractor = MLPFeatureExtractor(input_dim, dropout=dropout)
        self.label_predictor = BinaryLabelPredictor()
        self.domain_classifier = DomainClassifier(
            grl_alpha=grl_alpha,
            dropout=dropout,
        )

    def forward_label(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits = self.label_predictor(features)
        if return_features:
            return logits, features
        return logits

    def forward_domain(self, features):
        return self.domain_classifier(features)
