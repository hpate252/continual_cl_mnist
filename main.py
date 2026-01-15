import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SimpleCNN
from data_module import get_split_mnist_dataloaders
from ewc import compute_fisher_information, clone_params, ewc_loss
from replay import ReplayBuffer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device,
    criterion,
    method: str = "naive",
    replay_buffer: ReplayBuffer = None,
    replay_ratio: float = 0.5,
    prev_params: Dict[str, torch.Tensor] = None,
    fisher: Dict[str, torch.Tensor] = None,
    lambda_ewc: float = 0.0,
    update_buffer: bool = False,
) -> float:
    """
    Trains model for one epoch and returns average loss.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for inputs, targets in tqdm(dataloader, desc="Train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Add replay samples (for replay method during Task 2)
        if method == "replay" and replay_buffer is not None and len(replay_buffer) > 0:
            replay_bs = int(inputs.size(0) * replay_ratio)
            if replay_bs > 0:
                replay_x, replay_y = replay_buffer.sample(replay_bs)
                inputs = torch.cat([inputs, replay_x], dim=0)
                targets = torch.cat([targets, replay_y], dim=0)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Add EWC penalty if needed
        if method == "ewc" and prev_params is not None and fisher is not None:
            loss = loss + ewc_loss(model, prev_params, fisher, lambda_ewc)

        loss.backward()
        optimizer.step()

        # Fill replay buffer during Task 1 (if enabled)
        if update_buffer and replay_buffer is not None:
            replay_buffer.add_batch(inputs.detach(), targets.detach())

        running_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)

    return running_loss / max(1, n_samples)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device) -> Tuple[float, float]:
    """
    Returns (average loss, accuracy) on given dataloader.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc


def run_experiment(
    method: str = "naive",
    batch_size: int = 64,
    epochs_task1: int = 3,
    epochs_task2: int = 3,
    lambda_ewc: float = 500.0,
    replay_buffer_size: int = 2000,
    replay_ratio: float = 0.5,
    lr: float = 1e-3,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data for both tasks
    train_loader_t1, test_loader_t1 = get_split_mnist_dataloaders(
        task_id=1, batch_size=batch_size
    )
    train_loader_t2, test_loader_t2 = get_split_mnist_dataloaders(
        task_id=2, batch_size=batch_size
    )

    # Model & optimizer
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Optional replay buffer and EWC stats
    replay_buffer = (
        ReplayBuffer(capacity=replay_buffer_size, device=device)
        if method == "replay"
        else None
    )
    prev_params: Dict[str, torch.Tensor] = None
    fisher: Dict[str, torch.Tensor] = None

    print("\n=== Training on Task 1 (digits 0–4) ===")
    for epoch in range(epochs_task1):
        print(f"Epoch {epoch + 1}/{epochs_task1}")
        train_loss = train_one_epoch(
            model,
            train_loader_t1,
            optimizer,
            device,
            criterion,
            method="naive",  # Task 1 is always trained naively
            replay_buffer=replay_buffer,
            replay_ratio=replay_ratio,
            prev_params=None,
            fisher=None,
            lambda_ewc=0.0,
            update_buffer=(method == "replay"),
        )
        val_loss, val_acc = evaluate(model, test_loader_t1, device)
        print(
            f"Task1 Train Loss: {train_loss:.4f} | "
            f"Task1 Val Loss: {val_loss:.4f} | Task1 Val Acc: {val_acc:.2f}%"
        )

    # Snapshot for EWC
    if method == "ewc":
        print("\nComputing Fisher information on Task 1 for EWC...")
        fisher = compute_fisher_information(
            model, train_loader_t1, device=device, num_batches=100
        )
        prev_params = clone_params(model)
        print("Done computing Fisher.")

    print("\nAccuracy on Task 1 BEFORE learning Task 2:")
    _, acc_t1_before = evaluate(model, test_loader_t1, device)
    print(f"Task 1 Test Accuracy: {acc_t1_before:.2f}%")

    print("\n=== Training on Task 2 (digits 5–9) ===")
    for epoch in range(epochs_task2):
        print(f"Epoch {epoch + 1}/{epochs_task2}")
        train_loss = train_one_epoch(
            model,
            train_loader_t2,
            optimizer,
            device,
            criterion,
            method=method,
            replay_buffer=replay_buffer,
            replay_ratio=replay_ratio,
            prev_params=prev_params,
            fisher=fisher,
            lambda_ewc=lambda_ewc,
            update_buffer=False,  # Only fill buffer from Task 1
        )
        val_loss_t2, val_acc_t2 = evaluate(model, test_loader_t2, device)
        val_loss_t1, val_acc_t1 = evaluate(model, test_loader_t1, device)
        print(
            f"Task2 Train Loss: {train_loss:.4f} | "
            f"Task2 Val Loss: {val_loss_t2:.4f} | Task2 Val Acc: {val_acc_t2:.2f}% | "
            f"Task1 Val Acc (after Task2): {val_acc_t1:.2f}%"
        )

    print("\n=== Final Evaluation ===")
    loss_t1, acc_t1 = evaluate(model, test_loader_t1, device)
    loss_t2, acc_t2 = evaluate(model, test_loader_t2, device)

    print(f"Final Task 1 Test Loss: {loss_t1:.4f} | Task 1 Test Acc: {acc_t1:.2f}%")
    print(f"Final Task 2 Test Loss: {loss_t2:.4f} | Task 2 Test Acc: {acc_t2:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continual Learning on Split-MNIST with EWC and Replay."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="naive",
        choices=["naive", "ewc", "replay"],
        help="Training method: naive (fine-tuning), ewc, or replay.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size."
    )
    parser.add_argument(
        "--epochs-task1", type=int, default=3, help="Epochs on Task 1."
    )
    parser.add_argument(
        "--epochs-task2", type=int, default=3, help="Epochs on Task 2."
    )
    parser.add_argument(
        "--lambda-ewc",
        type=float,
        default=500.0,
        help="EWC regularization strength.",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=2000,
        help="Replay buffer capacity (number of samples).",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.5,
        help="Fraction of each batch drawn from replay buffer (for replay method).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda' or 'cpu'. If None, auto-detect.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        method=args.method,
        batch_size=args.batch_size,
        epochs_task1=args.epochs_task1,
        epochs_task2=args.epochs_task2,
        lambda_ewc=args.lambda_ewc,
        replay_buffer_size=args.replay_buffer_size,
        replay_ratio=args.replay_ratio,
        lr=args.lr,
        device=args.device,
    )
