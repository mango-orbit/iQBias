"""Dataset-backed audit records for the controlled BEFT Stage-A runs."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch


_MODES = (
    'linear', 'bitfit', 'q', 'k', 'v', 'v_out',
    'beft_v_norm', 'beft_v_norm_fc2',
    # Stage-D additions (pure BEFT-V + one extra single group, no VPTSP).
    'qv', 'v_fc2')

# Stage-D: modes whose fused-qkv mask selects more than one slice.
_MULTI_TARGET = {
    'qv': ('v', 'q'),
}

# Stage-D: explicit single-target table.  The legacy code derived the target
# from `self.mode[-1]`, which breaks for every Stage-D name whose last
# character is a digit ('v_fc2' -> '2', 'beft_v_norm_fc2' -> '2').
_QKV_TARGETS = {
    'q': ('q',),
    'k': ('k',),
    'v': ('v',),
    'v_out': ('v',),
    'beft_v_norm': ('v',),
    'beft_v_norm_fc2': ('v',),
    'qv': ('v', 'q'),
    'v_fc2': ('v',),
}


def _beft_qkv_targets(mode):
    """Slices of the fused qkv bias a mode is allowed to update."""
    if mode in _QKV_TARGETS:
        return _QKV_TARGETS[mode]
    if mode.startswith('beft_') and len(mode) > 5:
        return (mode[-1],)
    return ('v',)

# Stage-D: complete-bias extensions carried by a mode, beyond its qkv slice.
_EXTRA_BIAS_SUFFIXES = {
    'beft_v_norm': ('norm1.bias', 'norm2.bias'),
    'beft_v_norm_fc2': ('norm1.bias', 'norm2.bias', 'mlp.fc2.bias'),
    'v_fc2': ('mlp.fc2.bias',),
}


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.{}.tmp'.format(path.name))
    with open(temporary, 'w', encoding='utf-8') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2,
                  allow_nan=False)
        output.write('\n')
    os.replace(str(temporary), str(path))


def _atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.{}.tmp'.format(path.name))
    torch.save(payload, temporary)
    os.replace(str(temporary), str(path))


class StageABiasAudit:
    """Track metrics and prove that non-target backbone values stay exact."""

    def __init__(self, model, opt, output_path):
        self.network = model.module if hasattr(model, 'module') else model
        self.backbone = self.network.model
        self.opt = opt
        self.output_path = Path(output_path).expanduser().resolve()
        self.mode = str(getattr(opt, 'bias_mask', '')).lower()
        if self.mode not in _MODES:
            raise ValueError(
                'Stage-A audit requires an explicit bias mode, got {}.'.format(
                    self.mode))

        self.started_at = time.time()
        self.epoch_rows = []
        self.qkv_initial = []
        self.proj_initial = []
        for block in self.backbone.blocks:
            self.qkv_initial.append(block.attn.qkv.bias.detach().cpu().clone())
            self.proj_initial.append(block.attn.proj.bias.detach().cpu().clone())

        self.frozen_initial_digests = {}
        for name, parameter in self.backbone.named_parameters():
            if not self._parameter_may_change(name):
                self.frozen_initial_digests[name] = _tensor_digest(parameter)

        visible_total = sum(
            parameter.numel() for parameter in self.network.parameters()
            if parameter.requires_grad)
        visible_backbone = sum(
            parameter.numel() for parameter in self.backbone.parameters()
            if parameter.requires_grad)
        if self.mode.startswith('beft_'):
            counts = self.network.bias_parameter_accounting()
            effective_total = counts['effective_trainable_params']
            effective_backbone = counts['backbone_effective_trainable_params']
        else:
            effective_total = visible_total
            effective_backbone = visible_backbone
        self.parameter_counts = {
            'optimizer_visible_trainable_params': int(visible_total),
            'effective_trainable_params': int(effective_total),
            'backbone_optimizer_visible_trainable_params': int(
                visible_backbone),
            'backbone_effective_trainable_params': int(effective_backbone),
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(opt.device)
        self.payload = {
            'format_version': 1,
            'status': 'RUNNING',
            'method': self.mode,
            'seed': int(opt.seed),
            'dataset': opt.dataset,
            'pretrained_model': opt.pretrained_model,
            'n_epochs': int(opt.n_epochs),
            'amp': {
                'enabled': bool(opt.mix_precision),
                'dtype': opt.amp_dtype,
            },
            'parameter_counts': self.parameter_counts,
            'epochs': self.epoch_rows,
        }
        _atomic_json(self.output_path, self.payload)

    def _parameter_may_change(self, name):
        if self.mode == 'bitfit':
            return name.endswith('bias')
        if self.mode.startswith('beft_'):
            if name.endswith('attn.qkv.bias'):
                return True
            if (self.mode == 'v_out' and
                    name.endswith('attn.proj.bias')):
                return True
            if self.mode in ('beft_v_norm', 'beft_v_norm_fc2'):
                if (name.endswith('norm1.bias') or
                        name.endswith('norm2.bias') or name == 'norm.bias'):
                    return True
            if (self.mode == 'beft_v_norm_fc2' and
                    name.endswith('mlp.fc2.bias')):
                return True
            # Stage-D extensions.
            for suffix in _EXTRA_BIAS_SUFFIXES.get(self.mode, ()):
                if name.endswith(suffix):
                    return True
        return False

    @staticmethod
    def _metric(metrics, names):
        for values in metrics.values():
            for name in names:
                if name in values:
                    return float(values[name])
        return None

    def record_epoch(self, epoch, mean_train_loss, metrics, learning_rate,
                     runtime_seconds):
        row = {
            'epoch': int(epoch),
            'mean_train_loss': float(mean_train_loss),
            'r1': self._metric(metrics, ('e_recall@1', 'h_recall@1')),
            'r2': self._metric(metrics, ('e_recall@2', 'h_recall@2')),
            'r4': self._metric(metrics, ('e_recall@4', 'h_recall@4')),
            'map_at_r': self._metric(metrics, ('mAP_R', 'MAP@R', 'MAP')),
            'learning_rate': float(learning_rate),
            'runtime_seconds': float(runtime_seconds),
        }
        if not all(value is None or math.isfinite(value)
                   for value in row.values() if isinstance(value, float)):
            raise FloatingPointError(
                'Non-finite Stage-A metric at epoch {}.'.format(epoch))
        self.epoch_rows.append(row)
        _atomic_json(self.output_path, self.payload)

    def finalize(self):
        frozen_violations = []
        named_parameters = dict(self.backbone.named_parameters())
        for name, initial_digest in self.frozen_initial_digests.items():
            if _tensor_digest(named_parameters[name]) != initial_digest:
                frozen_violations.append(name)

        block_rows = []
        sum_squares = {'q': 0.0, 'k': 0.0, 'v': 0.0, 'out': 0.0}
        frozen_slice_violations = []
        for index, block in enumerate(self.backbone.blocks):
            final_qkv = block.attn.qkv.bias.detach().cpu()
            initial_qkv = self.qkv_initial[index]
            dim = int(final_qkv.numel() // 3)
            row = {'block': index}
            for slice_index, key in enumerate(('q', 'k', 'v')):
                start = slice_index * dim
                stop = start + dim
                delta = final_qkv[start:stop] - initial_qkv[start:stop]
                norm = float(torch.linalg.vector_norm(delta.float()).item())
                maximum = float(delta.abs().max().item())
                row['{}_delta_norm'.format(key)] = norm
                row['{}_max_abs_delta'.format(key)] = maximum
                sum_squares[key] += norm * norm
                beft_targets = _beft_qkv_targets(self.mode)
                if self.mode == 'linear' or (
                        self.mode.startswith('beft_') and
                        key not in beft_targets):
                    if maximum != 0.0:
                        frozen_slice_violations.append(
                            'blocks.{}.attn.qkv.bias.{}'.format(index, key))

            proj_delta = (
                block.attn.proj.bias.detach().cpu() - self.proj_initial[index])
            proj_norm = float(torch.linalg.vector_norm(
                proj_delta.float()).item())
            proj_maximum = float(proj_delta.abs().max().item())
            row['out_delta_norm'] = proj_norm
            row['out_max_abs_delta'] = proj_maximum
            sum_squares['out'] += proj_norm * proj_norm
            if self.mode in (
                    'linear', 'q', 'k', 'v', 'v_out',
                    'beft_v_norm', 'beft_v_norm_fc2',
                    'qv', 'v_fc2'):
                if proj_maximum != 0.0:
                    frozen_slice_violations.append(
                        'blocks.{}.attn.proj.bias'.format(index))
            block_rows.append(row)

        aggregate = {
            '{}_delta_norm'.format(key): math.sqrt(value)
            for key, value in sum_squares.items()
        }
        target_changed = True
        if self.mode in ('q', 'v'):
            target_changed = aggregate[
                '{}_delta_norm'.format(self.mode[-1])] > 0.0
        elif self.mode == 'v_out':
            target_changed = (
                aggregate['v_delta_norm'] > 0.0 and
                aggregate['out_delta_norm'] > 0.0)
        elif self.mode in ('beft_v_norm', 'beft_v_norm_fc2', 'v_fc2'):
            target_changed = aggregate['v_delta_norm'] > 0.0
        elif self.mode == 'qv':
            target_changed = (
                aggregate['v_delta_norm'] > 0.0 and
                aggregate['q_delta_norm'] > 0.0)
        # Attention K bias is mathematically cancelled by softmax translation;
        # a zero update is expected and is reported without turning the audit
        # into a false failure.

        all_epoch_metrics_present = (
            len(self.epoch_rows) == int(self.opt.n_epochs) and
            all(row['r1'] is not None and row['r2'] is not None and
                row['r4'] is not None and row['map_at_r'] is not None
                for row in self.epoch_rows))
        all_losses_finite = all(
            math.isfinite(row['mean_train_loss']) for row in self.epoch_rows)
        peak_bytes = (int(torch.cuda.max_memory_allocated(self.opt.device))
                      if torch.cuda.is_available() else 0)
        checks = {
            'all_epoch_metrics_present': all_epoch_metrics_present,
            'all_train_losses_finite': all_losses_finite,
            'frozen_parameters_exact': not frozen_violations,
            'frozen_qkv_slices_exact': not frozen_slice_violations,
            'target_changed_when_required': target_changed,
        }
        # Preserve exact float32 pre/final vectors.  The earlier ViT-S run
        # only retained final norms and best-epoch checkpoints, which made an
        # exact epoch-50 vector supplement impossible after the processes
        # exited.  This compact audit-side snapshot does not affect training.
        vector_path = self.output_path.with_name(
            'final_attention_bias_vectors.pt')
        vector_blocks = []
        for index, block in enumerate(self.backbone.blocks):
            vector_blocks.append({
                'block': int(index),
                'qkv_pre': self.qkv_initial[index].clone(),
                'qkv_post': block.attn.qkv.bias.detach().cpu().clone(),
                'out_pre': self.proj_initial[index].clone(),
                'out_post': block.attn.proj.bias.detach().cpu().clone(),
            })
        _atomic_torch_save(vector_path, {
            'format_version': 1,
            'method': self.mode,
            'epoch': int(self.opt.n_epochs),
            'pretrained_model': self.opt.pretrained_model,
            'blocks': vector_blocks,
        })
        self.payload.update({
            'status': 'PASS' if all(checks.values()) else 'FAIL',
            'runtime_seconds': float(time.time() - self.started_at),
            'peak_cuda_memory_bytes': peak_bytes,
            'qkv_delta_by_block': block_rows,
            'qkv_delta_aggregate': aggregate,
            'frozen_parameter_violations': frozen_violations,
            'frozen_slice_violations': frozen_slice_violations,
            'final_attention_bias_vectors': str(vector_path),
            'checks': checks,
        })
        _atomic_json(self.output_path, self.payload)
        return self.payload
