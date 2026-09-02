import torch
import torch.nn as nn
from torch.utils.data import Subset

from dengine.models.decorators import register_model
from dengine.models.utils import get_unique_targets
from dengine.models import ModuleBase


@register_model()
class UCI_HAR_US_Net(ModuleBase):
    """PyTorch implementation of the DeepConvLSTM architecture for Human Activity Recognition.

    References:
        - Ordóñez, F. J., & Roggen, D. (2016). Deep convolutional and LSTM recurrent
          neural networks for multimodal wearable activity recognition. Sensors, 16(1), 115.
        - Original implementation: https://github.com/STRCWearlab/DeepConvLSTM
        - Dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

    Notes:
        Originally ported from the Lasagne implementation using Gemini:
        https://github.com/STRCWearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    """

    def __init__(
        self,
        dataset: Subset,
        num_filters: int = 64,
        filter_size: int = 5,
        num_units_lstm: int = 128,
        dropout: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(dataset, *args, **kwargs)

        num_classes = len(get_unique_targets(dataset))

        # 4 successive 2D convolutions with kernel (filter_size, 1) and valid padding
        self.conv = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(filter_size, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=(filter_size, 1)),
            nn.ReLU(inplace=True),
        )

        # In Lasagne, DimshuffleLayer permutes to (batch, time, channels, sensors).
        # Flattening channels and sensor axes yields input_size = num_filters * nb_sensor_channels.
        # For UCI-HAR raw inertia (total_acc_xyz, body_acc_xyz, body_gyro_xyz), nb_sensor_channels = 9.
        sample_x, _ = dataset[0]
        # Handles shapes: (1, time, channels) or (time, channels)
        nb_sensor_channels = sample_x.shape[-1]
        lstm_in_features = num_filters * nb_sensor_channels

        self.lstm = nn.LSTM(
            input_size=lstm_in_features,
            hidden_size=num_units_lstm,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_units_lstm, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape normalization: expects (batch, 1, time, channels)
        if x.dim() == 3:
            # (batch, time, channels) -> (batch, 1, time, channels)
            x = x.unsqueeze(1)

        # Convolutions over the temporal axis (axis 2)
        out = self.conv(x)  # -> (batch, num_filters, time_reduced, channels)

        # Replicates Lasagne: DimshuffleLayer((0, 2, 1, 3)) + flattening trailing dims for LSTM
        # (batch, num_filters, time_reduced, channels) -> (batch, time_reduced, num_filters, channels)
        out = out.permute(0, 2, 1, 3).contiguous()
        batch_size, seq_len, num_filters, nb_channels = out.shape
        out = out.view(batch_size, seq_len, num_filters * nb_channels)

        # 2-layer LSTM
        lstm_out, _ = self.lstm(out)  # -> (batch, seq_len, num_units_lstm)

        # Replicates SliceLayer(..., -1, 1): take the final time-step of the window
        last_step = lstm_out[:, -1, :]  # -> (batch, num_units_lstm)
        last_step = self.dropout(last_step)

        # Linear projection yielding unnormalized logits (PyTorch CrossEntropyLoss expects raw logits)
        return self.fc(last_step)
