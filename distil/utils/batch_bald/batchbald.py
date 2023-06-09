"""
The following code is obtained from the BatchBALD Redux repository available
at https://github.com/BlackHC/batchbald_redux. Credit goes to Andreas Kirsch
and Joost van Amersfoort and Yarin Gal. If this code is used in a publication, 
please cite their original work:
    
@misc{kirsch2019batchbald,
    title={BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning},
    author={Andreas Kirsch and Joost van Amersfoort and Yarin Gal},
    year={2019},
    eprint={1906.08158},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}

Note that this file differs from the source in that the toma directives are removed; each received 
tensor should be moved to the correct device before calling these functions. Furthermore, the progress 
bars are removed.
"""

# AUTOGENERATED! DO NOT EDIT! File to edit: 01_batchbald.ipynb (unless otherwise specified).

__all__ = ['compute_conditional_entropy', 'compute_entropy', 'CandidateBatch', 'get_batchbald_batch', 'get_bald_batch']

# Cell
import math
from dataclasses import dataclass
from typing import List

import torch

from ..batch_bald import joint_entropy
import gc
# Cell


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    nats_N_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)

    entropies_N.copy_(-torch.sum(nats_N_K_C, dim=(1, 2)) / K)

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    mean_log_probs_N_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K)
    nats_N_C = mean_log_probs_N_C * torch.exp(mean_log_probs_N_C)

    entropies_N.copy_(-torch.sum(nats_N_C, dim=1))

    return entropies_N

# Cell


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def get_batchbald_batch(
    log_probs_N_K_C: torch.Tensor, batch_size: int, num_samples: int, dtype=None, device=None
) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return CandidateBatch(candidate_scores, candidate_indices)

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in range(batch_size):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())
    
    del batch_joint_entropy,scores_N, conditional_entropies_N
    gc.collect()
    torch.cuda.empty_cache()

    return CandidateBatch(candidate_scores, candidate_indices)

# Cell


def get_bald_batch(log_probs_N_K_C: torch.Tensor, batch_size: int, dtype=None, device=None) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    candiate_scores, candidate_indices = torch.topk(scores_N, batch_size)

    return CandidateBatch(candiate_scores.tolist(), candidate_indices.tolist())