from typing import Dict
import torch
import torch.nn as nn


@torch.no_grad()
def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Returns a dictionary of parameter_name -> detached clone of parameter tensor.
    Used to store θ* after Task 1.
    """
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def compute_fisher_information(
    model: nn.Module,
    dataloader,
    device,
    num_batches: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Approximates the diagonal Fisher information matrix for model parameters
    using gradients of the log-likelihood on data from dataloader.
    """
    model.eval()
    fisher = {
        name: torch.zeros_like(p, device=device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    n_batches = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        log_probs = torch.log_softmax(outputs, dim=1)
        # mean log-likelihood of the true class
        log_likelihood = log_probs[range(len(targets)), targets].mean()
        # we want gradient of *negative* log-likelihood
        (-log_likelihood).backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2

        n_batches += 1
        if n_batches >= num_batches:  # limit for speed
            break

    if n_batches == 0:
        return fisher

    for name in fisher:
        fisher[name] /= n_batches

    return fisher


def ewc_loss(
    model: nn.Module,
    prev_params: Dict[str, torch.Tensor],
    fisher: Dict[str, torch.Tensor],
    lambda_ewc: float,
) -> torch.Tensor:
    """
    Computes the EWC quadratic penalty:

      λ * Σ_i F_i * (θ_i - θ_i*)²
    """
    device = next(model.parameters()).device
    loss = torch.tensor(0.0, device=device)
    for name, p in model.named_parameters():
        if name in prev_params:
            loss += (fisher[name] * (p - prev_params[name]) ** 2).sum()
    return lambda_ewc * loss
