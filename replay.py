from typing import List, Tuple
import random
import torch


class ReplayBuffer:
    """
    Simple replay buffer that stores individual (image, label) pairs.
    Uses random replacement when capacity is exceeded.
    """

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.images: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []

    def __len__(self):
        return len(self.images)

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Store a batch of samples. We store them on CPU to save GPU memory.
        """
        # images: (B, C, H, W), labels: (B,)
        for img, label in zip(images, labels):
            img_cpu = img.detach().cpu()
            label_cpu = label.detach().cpu()

            if len(self.images) < self.capacity:
                self.images.append(img_cpu)
                self.labels.append(label_cpu)
            else:
                # Randomly replace an existing sample
                idx = random.randrange(self.capacity)
                self.images[idx] = img_cpu
                self.labels[idx] = label_cpu

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch from the buffer.
        """
        if len(self.images) == 0:
            raise ValueError("Replay buffer is empty.")
        indices = random.sample(
            range(len(self.images)), k=min(batch_size, len(self.images))
        )
        imgs = torch.stack([self.images[i] for i in indices]).to(self.device)
        labels = torch.stack([self.labels[i] for i in indices]).to(self.device)
        return imgs, labels
