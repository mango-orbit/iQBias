"""Semantic coding-rate utilities for selective BitFit experiments.

The calibration stage treats each backbone bias tensor as a structural group.
Fused QKV biases are split into virtual Q/K/V groups so the exact key-bias
invariance of standard self-attention can be audited and K can be excluded
without changing the model architecture.  Training manifests contain the
selected virtual groups and are validated against the pristine backbone bias
fingerprint before any optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import reduce

import torch
import torch.nn.functional as F

from criteria.rate_reduction import normalized_mcr2, normalized_proxy_rate


_QKV_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn\.qkv\.bias$")
_BLOCK_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


def _bias_type(name, component=None):
    lowered = name.lower()
    if component is not None:
        return "attn_{}".format(component)
    if "attn.proj.bias" in lowered:
        return "attn_proj"
    if "mlp.fc1.bias" in lowered:
        return "mlp_fc1"
    if "mlp.fc2.bias" in lowered:
        return "mlp_fc2"
    if "norm1.bias" in lowered:
        return "norm1"
    if "norm2.bias" in lowered:
        return "norm2"
    if lowered.endswith("norm.bias") or ".norm.bias" in lowered:
        return "final_norm"
    if "patch_embed" in lowered:
        return "patch_embed"
    return "other"


def describe_bias_groups(named_parameters):
    """Return JSON-serializable structural descriptors for backbone biases."""
    descriptors = []
    for name, parameter in named_parameters:
        if not name.endswith("bias"):
            continue
        flat_size = int(parameter.numel())
        qkv_match = _QKV_PATTERN.search(name)
        block_match = _BLOCK_PATTERN.search(name)
        block = int(block_match.group(1)) + 1 if block_match else None
        if qkv_match and parameter.ndim == 1 and flat_size % 3 == 0:
            width = flat_size // 3
            for index, component in enumerate(("q", "k", "v")):
                descriptors.append({
                    "group_id": "{}::{}".format(name, component),
                    "parameter_name": name,
                    "start": index * width,
                    "end": (index + 1) * width,
                    "numel": width,
                    "shape": list(parameter.shape),
                    "block": block,
                    "bias_type": _bias_type(name, component),
                    # K-bias adds the same scalar to every key logit for a
                    # fixed query and is therefore structurally ineligible.
                    "eligible": component != "k",
                })
        else:
            descriptors.append({
                "group_id": name,
                "parameter_name": name,
                "start": 0,
                "end": flat_size,
                "numel": flat_size,
                "shape": list(parameter.shape),
                "block": block,
                "bias_type": _bias_type(name),
                "eligible": True,
            })
    if not descriptors:
        raise ValueError("No backbone bias parameters were found.")
    return descriptors


def group_view(tensor, descriptor):
    """Return the flat tensor slice represented by one virtual bias group."""
    return tensor.reshape(-1)[descriptor["start"]:descriptor["end"]]


def backbone_bias_fingerprint(named_parameters):
    """Hash names, shapes and pristine values of every backbone bias."""
    digest = hashlib.sha256()
    found = False
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        if not name.endswith("bias"):
            continue
        found = True
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    if not found:
        raise ValueError("Cannot fingerprint a backbone without bias terms.")
    return digest.hexdigest()


def total_rate_score(features, eps_sq=0.5, jitter=1e-6):
    """Dimensionless total coding-rate score used by the UCR ablation."""
    _, score = normalized_proxy_rate(
        features, eps_sq=eps_sq, jitter=jitter)
    return score


def semantic_rate_score(features, labels, eps_sq=0.5, jitter=1e-6):
    """Dimensionless supervised MCR2 reduction used by SCR-BitFit."""
    _, _, _, score = normalized_mcr2(
        features, labels, eps_sq=eps_sq, jitter=jitter)
    return score


def _masked_logsumexp(values, mask, dim):
    minimum = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask, minimum)
    shape = list(values.shape)
    shape[dim] = 1
    zeros = torch.zeros(shape, device=values.device, dtype=values.dtype)
    return torch.logsumexp(torch.cat((masked, zeros), dim=dim), dim=dim)


def centroid_proxy_anchor_loss(features, labels, positive_alpha=32.0,
                               negative_alpha=32.0,
                               positive_delta=0.1,
                               negative_delta=-0.1):
    """Deterministic proxy-based task sensitivity on backbone features.

    Batch class centroids are detached proxies.  This avoids making the
    selection baseline depend on the random 512-D DML head and random learned
    proxies while retaining Proxy-Anchor's local semantic constraints.
    """
    labels = labels.reshape(-1).to(features.device, dtype=torch.long)
    features = F.normalize(features.float(), p=2, dim=-1)
    class_ids = torch.unique(labels, sorted=True)
    centroids = []
    for class_id in class_ids:
        centroids.append(features[labels == class_id].mean(dim=0))
    proxies = F.normalize(
        torch.stack(centroids, dim=0).detach(), p=2, dim=-1)
    similarities = features.matmul(proxies.transpose(0, 1))
    positive_mask = labels[:, None] == class_ids[None, :]
    negative_mask = ~positive_mask
    positive = -float(positive_alpha) * (
        similarities - float(positive_delta))
    negative = float(negative_alpha) * (
        similarities - float(negative_delta))
    return (_masked_logsumexp(positive, positive_mask, dim=0).mean() +
            _masked_logsumexp(negative, negative_mask, dim=0).mean())


class SensitivityAccumulator:
    """Accumulate non-cancelling per-batch RMS gradient statistics."""

    def __init__(self, descriptors):
        self.descriptors = list(descriptors)
        self._stats = {
            descriptor["group_id"]: {
                "sum_squared_norm": 0.0,
                "sum_batch_rms": 0.0,
                "sum_batch_rms_sq": 0.0,
                "batches": 0,
            }
            for descriptor in self.descriptors
        }

    def add(self, gradient_by_parameter):
        for descriptor in self.descriptors:
            gradient = gradient_by_parameter.get(
                descriptor["parameter_name"])
            if gradient is None:
                squared_norm = 0.0
            else:
                values = group_view(gradient.detach().float(), descriptor)
                squared_norm = float(values.square().sum().item())
            batch_rms = math.sqrt(
                squared_norm / max(int(descriptor["numel"]), 1))
            stats = self._stats[descriptor["group_id"]]
            stats["sum_squared_norm"] += squared_norm
            stats["sum_batch_rms"] += batch_rms
            stats["sum_batch_rms_sq"] += batch_rms * batch_rms
            stats["batches"] += 1

    def rows(self):
        output = []
        for descriptor in self.descriptors:
            stats = self._stats[descriptor["group_id"]]
            count = stats["batches"]
            if count <= 0:
                raise ValueError("Sensitivity accumulator is empty.")
            mean_rms = stats["sum_batch_rms"] / count
            mean_rms_sq = stats["sum_batch_rms_sq"] / count
            variance = max(mean_rms_sq - mean_rms * mean_rms, 0.0)
            std = math.sqrt(variance)
            score = math.sqrt(
                stats["sum_squared_norm"] /
                (count * max(int(descriptor["numel"]), 1)))
            row = dict(descriptor)
            row.update({
                "score": score,
                "mean_batch_rms": mean_rms,
                "std_batch_rms": std,
                "cv": std / max(mean_rms, 1e-30),
                "batches": count,
            })
            output.append(row)
        return output


class GradientAlignmentAccumulator:
    """Aggregate cosine between rate-loss and DML-loss gradients by group."""

    def __init__(self, descriptors):
        self.descriptors = list(descriptors)
        self._stats = {
            descriptor["group_id"]: {
                "rate_sq": 0.0, "dml_sq": 0.0, "dot": 0.0,
            }
            for descriptor in self.descriptors
        }

    def add(self, semantic_gradient, dml_gradient):
        for descriptor in self.descriptors:
            rate = semantic_gradient.get(descriptor["parameter_name"])
            dml = dml_gradient.get(descriptor["parameter_name"])
            if rate is None or dml is None:
                continue
            # Training would minimize -Phi_semantic, hence the minus sign.
            rate_values = -group_view(
                rate.detach().double(), descriptor)
            dml_values = group_view(dml.detach().double(), descriptor)
            stats = self._stats[descriptor["group_id"]]
            stats["rate_sq"] += float(rate_values.square().sum().item())
            stats["dml_sq"] += float(dml_values.square().sum().item())
            stats["dot"] += float((rate_values * dml_values).sum().item())

    def rows(self):
        output = []
        for descriptor in self.descriptors:
            stats = self._stats[descriptor["group_id"]]
            denominator = math.sqrt(stats["rate_sq"] * stats["dml_sq"])
            cosine = stats["dot"] / denominator if denominator > 0 else None
            row = dict(descriptor)
            row.update({
                "rate_loss_dml_cosine": cosine,
                "rate_gradient_norm": math.sqrt(stats["rate_sq"]),
                "dml_gradient_norm": math.sqrt(stats["dml_sq"]),
            })
            output.append(row)
        return output


def _knapsack_select(rows, budget):
    eligible = [row for row in rows if row.get("eligible", True)]
    if not eligible:
        raise ValueError("No eligible bias groups are available for selection.")
    costs = [int(row["numel"]) for row in eligible]
    cost_gcd = reduce(math.gcd, costs)
    capacity = int(budget) // cost_gcd
    # Each state is (value, tuple(group ids)); group counts are small for ViT.
    states = [None] * (capacity + 1)
    states[0] = (0.0, ())
    for row, raw_cost in zip(eligible, costs):
        cost = raw_cost // cost_gcd
        value = float(row["score"]) * raw_cost
        for used in range(capacity, cost - 1, -1):
            previous = states[used - cost]
            if previous is None:
                continue
            candidate = (previous[0] + value,
                         previous[1] + (row["group_id"],))
            current = states[used]
            if current is None or candidate[0] > current[0]:
                states[used] = candidate
    best_used, best = max(
        ((index, state) for index, state in enumerate(states)
         if state is not None),
        key=lambda item: (item[1][0], item[0]))
    return set(best[1]), best_used * cost_gcd


def select_groups(rows, budget_ratio):
    """Select structural groups under a true scalar-parameter budget."""
    ratio = float(budget_ratio)
    if not 0 < ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1].")
    eligible_total = sum(
        int(row["numel"]) for row in rows if row.get("eligible", True))
    budget = max(1, int(math.floor(eligible_total * ratio)))
    selected, selected_count = _knapsack_select(rows, budget)
    return {
        "budget_ratio": ratio,
        "budget_scalar_count": budget,
        "eligible_scalar_count": eligible_total,
        "selected_scalar_count": selected_count,
        "selected_group_ids": sorted(selected),
    }


def build_selection_manifest(objective, rows, budget_ratio, metadata=None):
    selection = select_groups(rows, budget_ratio)
    selected = set(selection["selected_group_ids"])
    manifest = {
        "format_version": 1,
        "method": "bias_group_selection",
        "objective": str(objective),
        **selection,
        "groups": [
            {**row, "selected": row["group_id"] in selected}
            for row in rows
        ],
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    return manifest


def save_json_atomic(payload, path):
    path = os.path.abspath(os.path.expanduser(str(path)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp.{}".format(os.getpid())
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, path)


def load_selection_manifest(path):
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            "Selective BitFit manifest does not exist: {}".format(resolved))
    with open(resolved, "r", encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported selective BitFit manifest version.")
    selected = manifest.get("selected_group_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Selective BitFit manifest selected no groups.")
    return manifest


def masks_from_manifest(named_parameters, manifest):
    """Create exact per-parameter masks and validate group descriptors."""
    named_parameters = list(named_parameters)
    descriptors = describe_bias_groups(named_parameters)
    descriptor_by_id = {
        descriptor["group_id"]: descriptor for descriptor in descriptors}
    selected = set(manifest["selected_group_ids"])
    unknown = selected.difference(descriptor_by_id)
    if unknown:
        raise ValueError(
            "Manifest contains unknown bias groups: {}".format(
                ", ".join(sorted(unknown))))
    parameters = dict(named_parameters)
    masks = {
        name: torch.zeros_like(parameter, memory_format=torch.preserve_format)
        for name, parameter in named_parameters if name.endswith("bias")
    }
    for group_id in selected:
        descriptor = descriptor_by_id[group_id]
        if not descriptor.get("eligible", True):
            raise ValueError(
                "Manifest illegally selected ineligible group {}.".format(
                    group_id))
        parameter = parameters[descriptor["parameter_name"]]
        if list(parameter.shape) != descriptor["shape"]:
            raise ValueError(
                "Bias shape changed for {}.".format(
                    descriptor["parameter_name"]))
        flat_mask = masks[descriptor["parameter_name"]].reshape(-1)
        flat_mask[descriptor["start"]:descriptor["end"]] = 1
    return descriptors, masks
