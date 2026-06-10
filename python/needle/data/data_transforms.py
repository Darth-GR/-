import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        flip_img = np.random.rand() < self.p
        return np.flip(img, axis=1).copy() if flip_img else img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        h, w, c = img.shape
        padded = np.pad(
            img,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            mode="constant",
        )
        start_x = self.padding + shift_x
        start_y = self.padding + shift_y
        return padded[start_x:start_x + h, start_y:start_y + w, :].copy()
