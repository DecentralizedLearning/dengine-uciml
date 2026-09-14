from typing import Tuple, Any

from torch import Tensor

import dengine.dataset


class SupervisedDataset(dengine.dataset.SupervisedDataset):
    def __init__(self, data: Tensor, targets: Tensor, subjects: Tensor, **kwargs):
        super().__init__(data, targets, **kwargs)
        self.subjects = subjects

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        return img, target
