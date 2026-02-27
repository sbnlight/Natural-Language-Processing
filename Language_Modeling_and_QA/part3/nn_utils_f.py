"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor
import torch.nn.functional as F  # [Added] Import functional interface for optimized kernels


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # [Optimization] Use PyTorch's optimized F.softmax.
    # It is significantly faster (C++/CUDA kernel) and handles numerical stability internally.
    return F.softmax(x, dim=dim)

    # --- Original Manual Implementation (Kept for reference) ---
    # Use the numerically stable implementation that handles all -inf rows
    # 1. Compute max for numerical stability
    # x_max = x.max(dim=dim, keepdim=True)[0]
    
    # 2. Handle rows where max is -inf (masked out entirely)
    # Replacing -inf with 0 avoids NaN during subtraction (-inf - -inf)
    # x_max_safe = x_max.masked_fill(x_max == float('-inf'), 0.0)
    
    # 3. Compute exponentials of shifted values
    # logits_shifted = x - x_max_safe
    # exps = torch.exp(logits_shifted)
    
    # 4. Compute sum
    # sum_exps = exps.sum(dim=dim, keepdim=True)
    
    # 5. Handle rows where sum is 0 (inputs were all -inf)
    # Replacing 0 with 1 avoids division by zero. 
    # Since the numerator exps are all 0 in this case, 0/1 = 0, which is correct.
    # sum_exps_safe = sum_exps.masked_fill(sum_exps == 0.0, 1.0)
    
    # return exps / sum_exps_safe


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # [Optimization] Use PyTorch's optimized F.cross_entropy.
    # This fuses log_softmax and nll_loss into a single kernel, avoiding
    # large intermediate tensor allocations (log_probs) and speeding up backward pass.
    return F.cross_entropy(logits, targets)

    # --- Original Manual Implementation (Kept for reference) ---
    # Numerically stable log_softmax: log(softmax(x)) = x - logsumexp(x)
    # logsumexp computes log(sum(exp(x))) in a stable way
    # log_sum_exp = torch.logsumexp(logits, dim=-1, keepdim=True)
    # log_probs = logits - log_sum_exp
    
    # Select the log probabilities corresponding to the target classes
    # N is the number of samples (or tokens)
    # N = logits.size(0)
    
    # Use advanced indexing to gather the log prob for the correct class for each sample
    # log_probs[i, targets[i]]
    # target_log_probs = log_probs[torch.arange(N, device=logits.device), targets]
    
    # Return the mean negative log likelihood
    # return -target_log_probs.mean()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    # [Optimization] Use torch.nn.utils.clip_grad_norm_.
    # This handles the calculation of global norm and scaling efficiently in C++.
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    # --- Original Manual Implementation (Kept for reference) ---
    # Filter parameters that have gradients
    # params = [p for p in parameters if p.grad is not None]
    
    # if not params:
    #     return torch.tensor(0.0)
    
    # Calculate the L2 global norm of all gradients concatenated
    # We compute the norm of each tensor, stack them, and compute the norm of that vector
    # device = params[0].grad.device
    # total_norm = torch.norm(
    #     torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in params]), 
    #     2
    # )
    
    # Calculate scaling coefficient
    # Add epsilon to avoid division by zero
    # clip_coef = max_norm / (total_norm + 1e-6)
    
    # Clamp coefficient at 1.0 to ensure we only scale down, never up
    # clip_coef = torch.clamp(clip_coef, max=1.0)
    
    # Apply clipping in-place
    # for p in params:
    #     p.grad.detach().mul_(clip_coef)
        
    # return total_norm


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    """
    # [Note] The original implementation is already reasonably vectorised.
    # Kept as is to ensure correct logic for masking.
    
    # Get predictions by finding the index with maximum logit score
    predictions = torch.argmax(logits, dim=-1)
    
    # Create mask for valid tokens (not ignore_index)
    mask = (targets != ignore_index)
    
    # Check correctness only on valid tokens
    # (predictions == targets) creates a boolean tensor
    # & mask ensures we only count matches where the target is valid
    correct = (predictions == targets) & mask
    
    # Count total valid tokens
    total_valid = mask.sum().float()
    
    # Handle edge case where there are no valid tokens
    if total_valid == 0:
        return torch.tensor(0.0, device=logits.device)
    
    # Compute accuracy
    accuracy = correct.sum().float() / total_valid
    return accuracy


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    """
    # [Optimization] Calculate perplexity using optimized cross_entropy.
    # Perplexity is simply exp(mean_cross_entropy).
    # F.cross_entropy with ignore_index automatically handles the masking and averaging
    # over valid tokens efficiently.
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return torch.exp(loss)

    # --- Original Manual Implementation (Kept for reference) ---
    # Compute log probabilities via stable log_softmax
    # log_sum_exp = torch.logsumexp(logits, dim=-1, keepdim=True)
    # log_probs = logits - log_sum_exp
    
    # Create mask for valid tokens
    # mask = (targets != ignore_index)
    
    # If no valid tokens, return a default value
    # if mask.sum() == 0:
    #     return torch.tensor(1.0, device=logits.device)

    # Safe targets for indexing: replace ignore_index with a valid index (e.g. 0).
    # We mask the results later, so the value at ignored positions doesn't matter,
    # but we need a valid index to avoid IndexError during gathering.
    # safe_targets = targets.masked_fill(~mask, 0)
    
    # Select log probs corresponding to targets
    # N = logits.size(0)
    # Using safe_targets prevents out-of-bounds access
    # target_log_probs = log_probs[torch.arange(N, device=logits.device), safe_targets]
    
    # Filter log probabilities to include only valid tokens
    # valid_log_probs = target_log_probs[mask]
        
    # Calculate mean negative log likelihood (cross-entropy) over valid tokens
    # mean_nll = -valid_log_probs.mean()
    
    # Perplexity = exp(cross_entropy)
    # return torch.exp(mean_nll)