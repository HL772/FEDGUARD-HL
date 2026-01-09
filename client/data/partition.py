from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir: str, download: bool = True) -> datasets.MNIST:
    transform = transforms.ToTensor()
    return datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)


def _dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    num_classes = int(labels.max()) + 1
    indices_per_class = [np.where(labels == cls)[0].tolist() for cls in range(num_classes)]

    for attempt in range(10):
        rng = np.random.default_rng(seed + attempt)
        shuffled_per_class = [cls_indices[:] for cls_indices in indices_per_class]
        for cls_indices in shuffled_per_class:
            rng.shuffle(cls_indices)

        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for cls_indices in shuffled_per_class:
            proportions = rng.dirichlet([alpha] * num_clients)
            counts = (proportions * len(cls_indices)).astype(int)
            diff = len(cls_indices) - counts.sum()
            if diff > 0:
                for idx in np.argsort(-proportions)[:diff]:
                    counts[idx] += 1
            start = 0
            for client_id, count in enumerate(counts):
                client_indices[client_id].extend(cls_indices[start : start + count])
                start += count

        if all(len(indices) > 0 for indices in client_indices):
            return client_indices

    return client_indices


def get_client_loader(
    client_rank: int,
    num_clients: int,
    alpha: float,
    seed: int,
    data_dir: str,
    batch_size: int,
    download: bool = True,
) -> Tuple[DataLoader, Dict[int, int]]:
    dataset = load_mnist(data_dir, download=download)
    labels = np.array(dataset.targets)
    partitions = _dirichlet_split(labels, num_clients, alpha, seed)
    client_indices = partitions[client_rank]
    subset = Subset(dataset, client_indices)
    label_hist: Dict[int, int] = {}
    for label in labels[client_indices]:
        label_hist[int(label)] = label_hist.get(int(label), 0) + 1
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader, label_hist
