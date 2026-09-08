import torch
import torch.nn as nn
from torch.utils.data import Subset

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets
from dengine.models import ModuleBase


@register_model()
class UCI_HAR_US_Net(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        hidden_dims: tuple = (512, 128),
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        # Targets are 0-5, so num_classes will dynamically evaluate to 6
        num_classes = len(get_unique_targets(dataset))

        # Dynamically grab the input feature size (should be 561)
        sample_x, _ = dataset[0]
        in_features = sample_x.shape[-1]

        layers = []
        prev_dim = in_features

        # Build hidden layers with Batch Normalization (highly recommended for
        # these hand-crafted statistical features) and Dropout
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final classification head yielding unnormalized logits
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
