import os
import pickle
from typing import Optional, List
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        for file_name in files:
            with open(os.path.join(base_folder, file_name), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            images.append(batch[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
            labels.extend(batch.get(b"labels", batch.get("labels")))
        self.X = np.concatenate(images, axis=0)
        self.y = np.array(labels, dtype=np.int8)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        image, label = self.X[index], self.y[index]
        if isinstance(index, (slice, list, np.ndarray)):
            return image, label
        if self.transforms is not None:
            hwc = image.transpose(1, 2, 0)
            hwc = self.apply_transforms(hwc)
            image = hwc.transpose(2, 0, 1)
        return image, label

    def __len__(self) -> int:
        return self.X.shape[0]
