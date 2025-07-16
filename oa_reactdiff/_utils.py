from typing import List

import math
import torch
from torch import Tensor
from torch_scatter import scatter, scatter_add, scatter_mean, segment_csr


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x

#def remove_mean_batch_deterministic(x, indices):
#    """Deterministic (drop-in replacement) version of remove_mean_batch.
#    """
#    device = x.device
#    indices = indices.to(device)
#    sorted_indices, sorted_perm = torch.sort(indices)
#    sorted_x = x[sorted_perm]
#
#    # Create indptr required by segment_csr.
#    num_segments = indices.max().item() + 1 if indices.numel() > 0 else 0
#    if num_segments == 0: # Handle empty indices
#        return x # No change if no batches
#    counts = torch.bincount(sorted_indices, minlength=num_segments).to(device)
#    indptr = torch.cat([torch.tensor([0], dtype=torch.long, device=device), counts.cumsum(dim=0)])
#    
#    mean_deterministic = segment_csr(src=sorted_x, indptr=indptr, reduce="mean")
#
#    # Re-map with indices to match the original order of x.
#    x_det = x - mean_deterministic[indices]
#    return x_det

def scatter_mean_deterministic(src, index, dim, dim_size=None):
    """Deterministic drop-in replacement for torch_scatter.scatter_mean(dim=0)."""
    if dim != 0:
        raise NotImplementedError("Deterministic scatter_mean only implemented for dim=0.")
    device = src.device
    index = index.to(device) 

    sorted_index, sorted_perm = torch.sort(index)
    sorted_src = src[sorted_perm]

    if dim_size is None:
        num_segments = index.max().item() + 1 if index.numel() > 0 else 0
    else:
        num_segments = dim_size

    if num_segments == 0:
        return torch.empty(0, *src.size()[1:], dtype=src.dtype, device=device)

    counts = torch.bincount(sorted_index, minlength=num_segments).to(device)
    indptr = torch.cat([torch.tensor([0], dtype=torch.long, device=device), counts.cumsum(dim=0)])

    output = segment_csr(src=sorted_src, indptr=indptr, reduce="mean")
    return output

def scatter_add_deterministic(src, index, dim, dim_size=None):
    """Deterministic drop-in replacement for torch_scatter.scatter_add(dim=0)."""
    if dim != 0:
        raise NotImplementedError("Deterministic scatter_add only implemented for dim=0.")

    device = src.device
    index = index.to(device) 

    sorted_index, sorted_perm = torch.sort(index)
    sorted_src = src[sorted_perm]

    if dim_size is None:
        num_segments = index.max().item() + 1 if index.numel() > 0 else 0
    else:
        num_segments = dim_size

    if num_segments == 0:
        return torch.empty(0, *src.size()[1:], dtype=src.dtype, device=device)

    counts = torch.bincount(sorted_index, minlength=num_segments).to(device)
    indptr = torch.cat([torch.tensor([0], dtype=torch.long, device=device), counts.cumsum(dim=0)])

    output = segment_csr(src=sorted_src, indptr=indptr, reduce="sum") # "sum" for addition instead of "mean".
    return output

def scatter_deterministic(src, index, dim, dim_size=None, reduce="sum"):
    """
    Deterministic drop-in replacement for torch_scatter.scatter(reduce="sum", dim=0).
    Only 'sum' reduction and dim=0 are implemented for deterministic behavior.
    """
    if dim != 0:
        raise NotImplementedError("Deterministic scatter only implemented for dim=0.")
    if reduce not in ["sum", "mean"]: # Extend if you need other deterministic reductions
        raise NotImplementedError(f"Deterministic scatter only implemented for 'sum' and 'mean' reduction, got {reduce}.")

    device = src.device
    index = index.to(device) 

    sorted_index, sorted_perm = torch.sort(index)
    sorted_src = src[sorted_perm]

    if dim_size is None:
        num_segments = index.max().item() + 1 if index.numel() > 0 else 0
    else:
        num_segments = dim_size

    if num_segments == 0:
        # Match torch_scatter's behavior for empty output when num_segments is 0
        return torch.empty(0, *src.size()[1:], dtype=src.dtype, device=device)

    counts = torch.bincount(sorted_index, minlength=num_segments).to(device)
    indptr = torch.cat([torch.tensor([0], dtype=torch.long, device=device), counts.cumsum(dim=0)])

    # Use segment_csr for the reduction
    output = segment_csr(src=sorted_src, indptr=indptr, reduce=reduce)
    return output


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter_add(x, node_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def sample_center_gravity_zero_gaussian_batch(
    size: List[int], indices: List[Tensor]
) -> Tensor:
    assert len(size) == 2
    x = torch.randn(size, device=indices[0].device)

    # This projection only works because Gaussian is rotation invariant
    # around zero and samples are independent!
    x_projected = remove_mean_batch(x, torch.cat(indices))
    return x_projected


def sum_except_batch(x, indices, dim_size):
    return scatter_add(x.sum(-1), indices, dim=0, dim_size=dim_size)

#def sum_except_batch_deterministic(x, indices, dim_size):
#    return scatter_add_deterministic(x.sum(-1), indices, dim=0, dim_size=dim_size)

def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def sample_gaussian(size, device):
    x = torch.randn(size, device=device)
    return x


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)

def set_deterministic_mode(enabled: bool):
    """
    Sets the deterministic mode for scatter operations globally within this module's scope.
    If enabled, replaces scatter_mean and scatter_add with their deterministic counterparts.
    """
    global scatter, scatter_mean, scatter_add # We need to make scatter_add also global here if we want to change it
    
    if enabled:
        print("Enabling deterministic mode for scatter operations.")
        scatter = scatter_deterministic
        scatter_mean = scatter_mean_deterministic
        scatter_add = scatter_add_deterministic
