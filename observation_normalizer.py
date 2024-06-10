"""
The file of ObservationNormalizer class
which makes an observation much easier for networks.
"""

import numpy as np


class ObservationNormalizer:
    """
    When an observation gets in, normalize it
    with mean and standard deviation of previous observations.
    """

    def __init__(self, s_dim):
        self.dim = s_dim
        self.s1_mean = np.zeros((s_dim), dtype=np.float32)
        self.s2_mean = np.zeros((s_dim), dtype=np.float32)
        self.s1_std = np.ones((s_dim), dtype=np.float32)
        self.s2_std = np.ones((s_dim), dtype=np.float32)

    def __call__(self, s1_np, s2_np=None):
        if s2_np is None:
            return (s1_np - self.s1_mean) / self.s1_std
        return (
            (s1_np - np.expand_dims(self.s1_mean, axis=(0, 1)))
            / np.expand_dims(self.s1_std, axis=(0, 1)),
            (s2_np - np.expand_dims(self.s2_mean, axis=(0, 1)))
            / np.expand_dims(self.s2_std, axis=(0, 1)),
        )

    def update(self, buffer):
        """
        Update ObservationNormalizer state.
        """
        s1_lst, s2_lst = [], []
        for data in buffer.buffer:
            s1_lst.append(data[0])
            s2_lst.append(data[1])
        s1_np = np.stack(s1_lst, axis=0)
        s2_np = np.stack(s2_lst, axis=0)
        self.s1_mean = np.mean(s1_np, axis=0)
        self.s2_mean = np.mean(s2_np, axis=0)
        self.s1_std = np.std(s1_np, axis=0) + 0.001
        self.s2_std = np.std(s2_np, axis=0) + 0.001
        return

    def save(self, path):
        """
        Save current ObservationNormalizer state.
        """
        total_np = np.concatenate(
            (self.s1_mean, self.s2_mean, self.s1_std, self.s2_std), axis=0
        )
        np.save(path, total_np)

    def load(self, path):
        """
        Load previous ObservationNormalizer state.
        """
        total_np = np.load(path)
        total_np = np.array(total_np, dtype=np.float32)
        self.s1_mean = total_np[: self.dim]
        self.s2_mean = total_np[self.dim : self.dim * 2]
        self.s1_std = total_np[self.dim * 2 : self.dim * 3]
        self.s2_std = total_np[self.dim * 3 :]
