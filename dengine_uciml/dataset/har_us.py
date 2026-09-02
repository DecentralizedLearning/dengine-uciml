from typing import List

import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from dengine.dataset.utils import filter_integer_targets, balanance_class_samples
from dengine.dataset.decorators import register_dataset

from .uci_dataset import SupervisedDataset


@register_dataset('uci_har_us')
def load_uci_har_us(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    subset_fraction: float = 1,
    *args, **kwargs
) -> SupervisedDataset:
    """https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"""
    har: List[np.ndarray] = fetch_openml(
        data_id=1478,
        data_home=output_path,
        as_frame=False,
        parser='auto',
        return_X_y=True,
    )  # type: ignore
    X, y = har
    y_int = y.astype(np.int64) - 1

    # Standard UCI-HAR train/test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=0.3, random_state=42, stratify=y_int
    )

    data = X_train if train else X_test
    targets = y_train if train else y_test

    assert 0 < subset_fraction <= 1
    end = int(len(data) * subset_fraction)

    data_tensor = torch.as_tensor(data[:end], dtype=torch.float32)
    targets_tensor = torch.as_tensor(targets[:end], dtype=torch.int64)

    dataset = SupervisedDataset(
        data=data_tensor,
        targets=targets_tensor,
        transform=None,
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset

    return filter_integer_targets(dataset, target_labels)
