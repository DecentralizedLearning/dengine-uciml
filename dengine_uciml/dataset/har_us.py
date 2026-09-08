from typing import List
import os
import zipfile
import shutil
import urllib.request

import torch
from tqdm import tqdm
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from dengine.dataset.utils import filter_integer_targets, balanance_class_samples
from dengine.dataset.decorators import register_dataset

from .uci_dataset import SupervisedDataset


@register_dataset('openml_uci_har_us')
def load_openml_uci_har_us(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    subset_fraction: float = 1,
    *args, **kwargs
) -> SupervisedDataset:
    """https://www.openml.org/search?type=data&sort=runs&id=1478&status=active"""
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


# ##################################### #
# Official uci archive.
# As of today (8/9/2026) the dataset is not supported natively by https://github.com/uci-ml-repo/ucimlrepo
# The following script fetches the data automatically and unzip the dataset
# ##################################### #
_DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
_OUTER_ZIP_NAME = "human+activity+recognition+using+smartphones.zip"
_INNER_ZIP_NAME = "UCI HAR Dataset.zip"
_DATA_DIR_NAME = "UCI HAR Dataset"


def _download_zip(url: str, dest_path: str) -> None:
    """Download url to dest_path atomically, validating the result is a real zip."""
    tmp_path = dest_path + ".part"

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path) as pbar:
        def reporthook(block_num, block_size, total_size):
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urllib.request.urlretrieve(url, tmp_path, reporthook)

    if not zipfile.is_zipfile(tmp_path):
        os.remove(tmp_path)
        raise zipfile.BadZipFile(
            f"Downloaded file from {url} is not a valid zip "
            f"(server may have returned an error page)"
        )

    shutil.move(tmp_path, dest_path)


def _download_and_extract(output_path: str) -> str:
    """Ensure the UCI HAR dataset is downloaded and extracted under output_path.

    Returns the path to the extracted 'UCI HAR Dataset' directory.
    """
    os.makedirs(output_path, exist_ok=True)

    data_dir = os.path.join(output_path, _DATA_DIR_NAME)
    if os.path.isdir(data_dir) and os.path.exists(
        os.path.join(data_dir, "train", "X_train.txt")
    ):
        return data_dir

    outer_zip_path = os.path.join(output_path, _OUTER_ZIP_NAME)

    # Remove a stale/corrupt cached file from a previous failed run, if any.
    if os.path.exists(outer_zip_path) and not zipfile.is_zipfile(outer_zip_path):
        os.remove(outer_zip_path)

    if not os.path.exists(outer_zip_path):
        _download_zip(_DATASET_URL, outer_zip_path)

    with zipfile.ZipFile(outer_zip_path, "r") as zf:
        zf.extractall(output_path)

    inner_zip_path = os.path.join(output_path, _INNER_ZIP_NAME)

    if not zipfile.is_zipfile(inner_zip_path):
        raise RuntimeError(
            f"Extracted inner archive at {inner_zip_path} is not a valid zip; "
            f"try deleting {output_path} and re-running"
        )

    with zipfile.ZipFile(inner_zip_path, "r") as zf:
        zf.extractall(output_path)

    return data_dir


def _load_split(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the X/y arrays for 'train' or 'test' from the extracted dataset."""
    split_dir = os.path.join(data_dir, split)
    X = np.loadtxt(os.path.join(split_dir, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(split_dir, f"y_{split}.txt"))
    return X, y


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
    data_dir = _download_and_extract(output_path)

    split = "train" if train else "test"
    X, y = _load_split(data_dir, split)

    # UCI-HAR labels are 1-indexed
    y_int = y.astype(np.int64) - 1

    assert 0 < subset_fraction <= 1
    end = int(len(X) * subset_fraction)

    data_tensor = torch.as_tensor(X[:end], dtype=torch.float32)
    targets_tensor = torch.as_tensor(y_int[:end], dtype=torch.int64)

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
