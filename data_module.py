from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _split_indices_by_labels(dataset, allowed_labels: List[int]):
    """
    Returns indices of samples whose label is in allowed_labels.
    """
    indices = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if int(label) in allowed_labels:
            indices.append(idx)
    return indices


def get_split_mnist_dataloaders(
    task_id: int,
    batch_size: int = 64,
    data_root: str = "./data",
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and test dataloaders for a specific Split-MNIST task.

    Task 1: digits 0–4
    Task 2: digits 5–9
    """
    assert task_id in (1, 2), "Only task_id 1 or 2 supported."

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Standard MNIST mean/std
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    if task_id == 1:
        allowed = [0, 1, 2, 3, 4]
    else:
        allowed = [5, 6, 7, 8, 9]

    train_indices = _split_indices_by_labels(train_dataset, allowed)
    test_indices = _split_indices_by_labels(test_dataset, allowed)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
