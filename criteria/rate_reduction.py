"""Exact, numerically stable coding-rate primitives for downstream DML.

The convention follows MCR2 and Anti-Collapse Loss: ``eps_sq`` denotes
epsilon squared, so the matrix multiplier is ``d / (m * eps_sq)``.
All log-determinants are evaluated in FP32 outside autocast.  Sylvester's
identity lets us choose the smaller of the sample and feature Gram matrices
without changing the result.
"""

import math

import torch
import torch.nn.functional as F


def _half_logdet_spd(matrix, jitter=1e-6):
    """Return 0.5 log(det(matrix)) for a symmetric positive matrix."""
    chol, info = torch.linalg.cholesky_ex(matrix)
    if torch.any(info):
        identity = torch.eye(
            matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        chol, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
    if torch.any(info):
        failed = int(torch.count_nonzero(info).item())
        raise RuntimeError(
            'Coding-rate Cholesky failed for {} matrix/matrices; increase '
            '--loss_ratefit_jitter.'.format(failed))
    return chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)


def coding_rate(features, eps_sq=0.5, jitter=1e-6,
                normalize_features=True):
    r"""Compute ``0.5 logdet(I + d/(m eps^2) Z Z^T)`` exactly."""
    if features.ndim != 2:
        raise ValueError(
            'Coding-rate features must have shape [samples, dimensions], got '
            '{}.'.format(tuple(features.shape)))
    m, d = features.shape
    if m < 1 or d < 1:
        raise ValueError('Coding rate requires non-empty features.')
    if eps_sq <= 0:
        raise ValueError('eps_sq must be greater than zero.')
    if jitter < 0:
        raise ValueError('jitter must be non-negative.')

    # float() alone is insufficient when an outer CUDA autocast context is
    # active, so explicitly disable it around every matrix multiplication.
    with torch.cuda.amp.autocast(enabled=False):
        z = features.float()
        if normalize_features:
            z = F.normalize(z, p=2, dim=-1)
        scale = float(d) / (float(m) * float(eps_sq))
        if m <= d:
            gram = z.matmul(z.transpose(0, 1))
            matrix_size = m
        else:
            gram = z.transpose(0, 1).matmul(z)
            matrix_size = d
        identity = torch.eye(
            matrix_size, dtype=z.dtype, device=z.device)
        return _half_logdet_spd(identity + scale * gram, jitter=jitter)


def mcr2_rate_reduction(features, labels, eps_sq=0.5, jitter=1e-6):
    r"""Return exact supervised MCR2 expansion, compression and reduction.

    ``conditional_rate`` implements
    ``sum_c (m_c / m) R(Z_c, eps)`` rather than treating compression as an
    informal similarity penalty.
    """
    labels = labels.reshape(-1).to(device=features.device, dtype=torch.long)
    if labels.numel() != features.shape[0]:
        raise ValueError(
            'MCR2 received {} labels for {} features.'.format(
                labels.numel(), features.shape[0]))
    total_count = int(features.shape[0])
    total_rate = coding_rate(features, eps_sq=eps_sq, jitter=jitter)
    conditional_rate = total_rate.new_zeros(())
    for class_id in torch.unique(labels, sorted=True):
        class_features = features[labels == class_id]
        class_weight = float(class_features.shape[0]) / float(total_count)
        conditional_rate = conditional_rate + class_weight * coding_rate(
            class_features, eps_sq=eps_sq, jitter=jitter)
    return total_rate, conditional_rate, total_rate - conditional_rate


def normalized_proxy_rate(proxies, eps_sq=0.5, jitter=1e-6):
    r"""Normalize proxy coding rate by its unit-vector orthogonal maximum.

    For ``k <= d`` unit proxies, the maximum is
    ``k/2 * log(1 + d/(k eps^2))``.  The resulting score is dimensionless and
    is approximately bounded by one, making its loss weight portable.
    """
    k, d = proxies.shape
    if k < 1:
        raise ValueError('At least one active proxy is required.')
    rate = coding_rate(proxies, eps_sq=eps_sq, jitter=jitter)
    if k <= d:
        maximum = 0.5 * float(k) * math.log1p(
            float(d) / (float(k) * float(eps_sq)))
    else:
        # This branch is not expected for CUB batches, but provides the dual
        # tight-frame upper bound for completeness.
        maximum = 0.5 * float(d) * math.log1p(1.0 / float(eps_sq))
    return rate, rate / max(maximum, 1e-12)


def normalized_mcr2(features, labels, eps_sq=0.5, jitter=1e-6):
    """Return MCR2 terms plus a scale-free reduction score.

    The denominator is detached deliberately: it changes only the numerical
    scale of the MCR2 gradient and cannot be optimized as a shortcut.
    """
    total_rate, conditional_rate, reduction = mcr2_rate_reduction(
        features, labels, eps_sq=eps_sq, jitter=jitter)
    score = reduction / total_rate.detach().clamp_min(1e-12)
    return total_rate, conditional_rate, reduction, score
