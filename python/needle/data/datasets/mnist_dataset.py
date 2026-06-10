import struct
import gzip
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


def parse_mnist(image_filesname, label_filename):
    with gzip.open(image_filesname, "rb") as img_f:
        magic, num_images, rows, cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"invalid MNIST image file magic number: {magic}")
        images = np.frombuffer(img_f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
    with gzip.open(label_filename, "rb") as label_f:
        magic, num_labels = struct.unpack(">II", label_f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label file magic number: {magic}")
        labels = np.frombuffer(label_f.read(), dtype=np.uint8)
    if num_images != num_labels:
        raise ValueError("MNIST image and label counts differ")
    return images.astype(np.float32) / 255.0, labels.astype(np.int8)


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        image = self.images[index]
        if isinstance(index, (slice, list, np.ndarray)):
            return image, self.labels[index]
        image = self.apply_transforms(image.reshape(28, 28, 1)).reshape(-1)
        return image, self.labels[index]

    def __len__(self) -> int:
        return self.images.shape[0]
