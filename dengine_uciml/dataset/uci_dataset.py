from typing import Tuple, Any

import dengine.dataset


class SupervisedDataset(dengine.dataset.SupervisedDataset):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        return img, target
