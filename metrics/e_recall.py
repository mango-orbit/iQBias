import numpy as np


class Metric():
    def __init__(self, k, **kwargs):
        self.k = k
        self.requires = ['nearest_features@{}'.format(k), 'target_labels']
        self.name = 'e_recall@{}'.format(k)

    def __call__(self, target_labels, k_closest_classes):
        targets = np.asarray(target_labels).reshape(-1, 1)
        recalled = np.asarray(k_closest_classes)[:, :self.k]
        return np.any(recalled == targets, axis=1).mean()
