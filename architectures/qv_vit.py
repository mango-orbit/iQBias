import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
import contextlib
import copy
import inspect
import json
import os
import re

import timm
from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Attention, LayerScale
import math
import numpy as np

from utilities.bias_groups import (
    backbone_bias_fingerprint,
    describe_bias_groups,
    group_view,
    load_selection_manifest,
    masks_from_manifest,
)
from utilities.hyperbolic import ToPoincare


BIAS_MASK_MODES = ('q', 'k', 'v', 'v_out',
                  'qv', 'v_fc2')

# Stage-D extensions: pure BEFT-V combined with ONE extra single group,
# WITHOUT VPTSP.  Maps mode -> (extra fused-qkv targets, include_norm,
# include_fc2).  The legacy modes are absent here and keep the original
# expression, so their behaviour is bit-identical.
BIAS_MASK_EXTENSIONS = {
    'qv': (('q',), False, False),
    'v_fc2': ((), False, True),
}


def parse_bias_layers(value, block_depth):
    """Normalize the Stage-C BEFT block selector without changing BEFT math."""
    if value is None or (isinstance(value, str) and value.strip().lower() == 'all'):
        return tuple(range(int(block_depth)))
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]
        try:
            layers = [int(item.strip()) for item in text.split(',')
                      if item.strip()]
        except ValueError as error:
            raise ValueError('bias_layers must be all or integer indices.') from error
    elif isinstance(value, (list, tuple)):
        if any(isinstance(item, bool) for item in value):
            raise ValueError('bias_layers does not accept booleans.')
        layers = [int(item) for item in value]
    else:
        raise ValueError('bias_layers must be all or an explicit list.')
    if not layers:
        raise ValueError('bias_layers explicit list cannot be empty.')
    if len(set(layers)) != len(layers):
        raise ValueError('bias_layers contains duplicate indices.')
    invalid = [index for index in layers
               if index < 0 or index >= int(block_depth)]
    if invalid:
        raise ValueError(
            'bias_layers indices out of range [0, {}]: {}.'.format(
                int(block_depth) - 1, invalid))
    return tuple(sorted(layers))


def audit_attention_qkv_layout(attention, embed_dim=None):
    """Verify that a timm Attention uses one fused [Q, K, V] projection.

    BEFT slicing is unsafe if inferred from the parameter shape alone.  This
    audit therefore checks both the module structure and the installed
    Attention.forward source before returning the slice boundaries.
    """
    if not hasattr(attention, 'qkv') or not isinstance(attention.qkv, nn.Linear):
        raise RuntimeError('Attention does not expose a fused nn.Linear qkv.')
    if attention.qkv.bias is None:
        raise RuntimeError('Attention qkv projection has no bias parameter.')
    dim = int(embed_dim if embed_dim is not None else attention.qkv.in_features)
    if attention.qkv.in_features != dim or attention.qkv.out_features != 3 * dim:
        raise RuntimeError(
            'Expected qkv Linear({}, {}), got Linear({}, {}).'.format(
                dim, 3 * dim, attention.qkv.in_features,
                attention.qkv.out_features))
    if attention.qkv.bias.numel() != 3 * dim:
        raise RuntimeError(
            'Expected fused qkv.bias with {} elements, got {}.'.format(
                3 * dim, attention.qkv.bias.numel()))

    try:
        source = inspect.getsource(attention.forward)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            'Cannot inspect Attention.forward; refusing to assume Q/K/V '
            'layout.') from error
    compact = re.sub(r'\s+', '', source)
    reshape_marker = re.search(
        r'qkv=self\.qkv\(x\)\.reshape\([^\n]*?,3,', compact)
    unbind_marker = 'q,k,v=qkv.unbind(0)' in compact
    if reshape_marker is None or not unbind_marker:
        raise RuntimeError(
            'Attention.forward was not recognized as fused '
            'reshape(..., 3, ...) followed by q,k,v=qkv.unbind(0); '
            'refusing to apply a three-way mask.')

    return {
        'layout': 'qkv',
        'embed_dim': dim,
        'slices': {
            'q': (0, dim),
            'k': (dim, 2 * dim),
            'v': (2 * dim, 3 * dim),
        },
        'qk_norm': not (
            isinstance(getattr(attention, 'q_norm', None), nn.Identity) and
            isinstance(getattr(attention, 'k_norm', None), nn.Identity)),
        'fused_attn': bool(getattr(attention, 'fused_attn', False)),
        'forward_class': '{}.{}'.format(
            attention.__class__.__module__, attention.__class__.__name__),
    }

# How the prompt/proxy code in this file is organised:
#   classic proxy : one randomly initialised learnable vector per class.
#   shared prompt : the frozen ViT is adapted for sample encoding by
#                   shared visual prompts.
#   proxy prompt  : class-conditioned prompts generate candidate proxies
#                   that carry image semantics.
#   accumulator   : EMA or GRU accumulation of those candidates across
#                   samples and batches.
#   fusion        : accumulated semantic proxies are mixed with the
#                   learnable proxy vector, named proxy_bias here.
#   prompt bank   : the full class-prompt library stays on the CPU and
#                   only the current batch moves to a GPU buffer.
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        
        self.pars = opt
        self.hyp_metric_enabled = bool(getattr(opt, 'hyp_metric', False) or opt.loss == 'hyp_pairwise')
        self.name = opt.pretrained_model
        
        self.model = self._build_backbone(opt)
        self.rate_projection_enabled = (
            getattr(opt, 'rate_projection_mode', 'none') != 'none')
        self.reference_geometry_enabled = opt.loss in (
            'prrd_proxy_peft', 'relation_kd_proxy_peft',
            'relation_frob_budget_proxy_peft',
            'head_relation_budget_proxy_peft') or self.rate_projection_enabled
        if self.reference_geometry_enabled:
            # The teacher is the exact pristine backbone.  It is a registered
            # frozen submodule so model.to(device) moves it safely, but it is
            # never placed in an optimizer group or serialized as a trainable
            # PEFT checkpoint.
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.requires_grad_(False)
            self.reference_model.eval()
        self.embed_dim = self.model.embed_dim 
        self.block_depth = len(self.model.blocks)
        # default_cfg keeps the pretraining resolution even when img_size is
        # overridden, so use the actual patch embed input size here.
        patch_height, patch_width = self.model.patch_embed.img_size
        self.input_size = (3, patch_height, patch_width)
        self.p_channels = self._cal_channels()
        self.n_classes = opt.n_classes # 
        self.distill = ('distilled' in opt.pretrained_model)
        # n_classes_batch is the number of classes in a batch
        self.n_classes_batch = opt.batch_size // opt.samples_per_class

        model_params = []
        proxy_net_params = []
        prompt_params = []
        proxy_prompt_params = []
        bitfit_params = []
        self.optimizer = None
        self.bitfit_selection_manifest = None
        self.bitfit_selection_manifest_path = ''
        self.bitfit_optimizer_bias_count = 0
        self.bitfit_effective_bias_count = 0
        self.bitfit_eligible_bias_count = 0
        self._selective_restore_entries = []
        # [patched] per-tensor bool mask of the BEFT slice, so that a composed
        # SCR restore pass widens its exclusive mask to the union instead of
        # reverting the BEFT slice to pristine every step.
        self._beft_slice_mask_by_name = {}
        # [patched] bare param name -> SCR mask buffer name, so the
        # BEFT restore pass can preserve the SCR group as well.
        self._scr_mask_name_by_bare_name = {}
        self._scr_beft_hook_unioned = False
        self._beft_hook_unioned = False
        self._bitfit_group_descriptors = []
        self._bitfit_initial_values = {}
        self._bitfit_selection_masks = {}
        self._bitfit_calibration_by_group = {}
        self.bitfit_gain_manifest = None
        self.bitfit_gain_manifest_path = ''
        self._gain_scale_buffer_by_parameter = {}
        self.bias_mask = str(getattr(
            self.pars, 'bias_mask', 'auto')).lower()
        if self.bias_mask not in (
                'auto', 'linear', 'bitfit') + BIAS_MASK_MODES:
            raise ValueError(
                'Unsupported bias_mask: {}'.format(
                    self.bias_mask))
        self._beft_restore_entries = []
        self._beft_hook_handles = []
        self.beft_layout_audit = []
        self.beft_optimizer_backbone_count = 0
        self.beft_effective_backbone_count = 0
        self.bias_selected_layers = parse_bias_layers(
            getattr(self.pars, 'bias_layers', 'all'), self.block_depth)
        self.bias_layers = (
            'all' if self.bias_selected_layers == tuple(range(self.block_depth))
            else list(self.bias_selected_layers))
        if (self.bias_layers != 'all' and
                self.bias_mask != 'v'):
            raise ValueError(
                'Explicit bias_layers is supported only with v.')

        # pretrained configuration
        self.pretrained_cfg = self.model.pretrained_cfg

        # Unlike the full fine-tuning proxy-anchor baseline, the pretrained ViT
        # is frozen here and only the adaptation modules selected below are
        # optimised: the projection head, the prompts and the bias slices.
        if not self.pars.full_fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            if self.bias_mask == 'linear':
                # Explicit linear mode never trains a backbone bias, even if
                # the caller independently enables full weight fine-tuning.
                for name, param in self.model.named_parameters():
                    if name.endswith('bias'):
                        param.requires_grad = False
                    else:
                        model_params.append(param)
            else:
                model_params += list(self.model.parameters())

        ###################### All trainable setting must be below this line ######################

        # setting head
        if self.pars.train_head:
            if self.pars.pooling == 'combined':
                self.head, self.fc_norm, self.dist_head, self.dist_norm = self._setup_head(in_dim=self.embed_dim*2)
            else:
                self.head, self.fc_norm, self.dist_head, self.dist_norm = self._setup_head()
            model_params += list(self.head.parameters())
            model_params += list(self.fc_norm.parameters())
            if self.distill:
                model_params += list(self.dist_head.parameters())
                model_params += list(self.dist_norm.parameters())

        self.dropout = nn.Dropout(self.pars.feature_dropout)
        self.hyp_projection = ToPoincare(getattr(opt, 'hyp_c', 0.1), getattr(opt, 'hyp_clip_r', 2.3))

        # O_c / proxies_bias: the classic randomly initialised learnable class
        # proxy, retained as the proxy-bias term of the fusion.
        # P_c / proxies_container: semantic state produced by the image and the
        # class prompts and accumulated over batches. It is cross-batch state,
        # not an optimiser-updated Parameter.
        self.proxies_bias = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros(self.n_classes, self.pars.embed_dim)
           ) / 8.0)

        self.proxies_container = nn.init.xavier_uniform_(
                        torch.zeros(self.n_classes,
                        self.pars.embed_dim).to(self.pars.device) / 8.0)

        self.proxies_container = F.normalize(self.proxies_container, dim=1)

        # Learnable recurrent semantic accumulator. The alternative
        # semantic_mix_type="mix" corresponds to an EMA-style update.
        if "gru" in self.pars.semantic_mix_type and self.pars.semantic_proxy:
            self.rnn_update = Simple_GRUCell(self.pars.embed_dim, self.pars.embed_dim,
                                             drop=self.pars.rnn_dropout, type=self.pars.semantic_mix_type)
            proxy_net_params += list(self.rnn_update.parameters())


        if self.pars.adapter:
            model_params = self._setup_adapter(model_params)

        # The semantic-proxy branch uses its own projection head and does not
        # share the DML projection head of the sample-encoding branch.
        if self.pars.semantic_proxy:
            if self.pars.pooling == 'combined':
                self.p_head, self.p_fc_norm, self.p_dist_head, self.p_dist_norm = self._setup_head(in_dim=self.embed_dim*2)
            else:
                self.p_head, self.p_fc_norm, self.p_dist_head, self.p_dist_norm = self._setup_head()

            proxy_net_params += list(self.p_head.parameters())
            proxy_net_params += list(self.p_fc_norm.parameters())
            if self.distill:
                proxy_net_params += list(self.p_dist_head.parameters())
                proxy_net_params += list(self.p_dist_norm.parameters())

        # setting bitfit
        bitfit_lr_scale = float(getattr(
            self.pars, 'bitfit_lr_scale', 1.0))
        if bitfit_lr_scale < 0:
            raise ValueError('bitfit_lr_scale must be non-negative.')
        if self.pars.full_fine_tune and bitfit_lr_scale != 1.0:
            raise ValueError(
                'bitfit_lr_scale != 1 is incompatible with full_fine_tune: '
                'backbone biases would occur in two optimizer groups.')
        selective_bitfit = bool(getattr(
            self.pars, 'vit_selective_bitfit', False))
        gainnorm_bitfit = bool(getattr(
            self.pars, 'vit_gainnorm_bitfit', False))
        headaware_gainnorm_bitfit = bool(getattr(
            self.pars, 'vit_headaware_gainnorm_bitfit', False))
        legacy_standard_bitfit = bool(
            self.pars.all_bias_bitfit or
            getattr(self.pars, 'vit_rc_bitfit', False) or
            gainnorm_bitfit or headaware_gainnorm_bitfit)
        standard_bitfit = (
            legacy_standard_bitfit if self.bias_mask == 'auto'
            else self.bias_mask == 'bitfit')
        bias_mask_mode = self.bias_mask in BIAS_MASK_MODES
        # See main.py: selective BitFit may be composed with an explicit BEFT
        # mode (SCR manifest = extra groups, BEFT = its own slice).
        selective_with_beft = bool(
            selective_bitfit and self.bias_mask in BIAS_MASK_MODES)
        if (self.bias_mask != 'auto' and
                ((selective_bitfit and not selective_with_beft) or
                 getattr(self.pars, 'vit_rc_bitfit', False)
                 or gainnorm_bitfit or headaware_gainnorm_bitfit)):
            raise ValueError(
                'Explicit bias_mask cannot be combined with selective, '
                'RC, GainNorm, or Head-aware GainNorm BitFit modes.')
        if bias_mask_mode and self.pars.full_fine_tune:
            raise ValueError(
                'BEFT requires a frozen backbone; full_fine_tune is active.')
        if selective_bitfit and standard_bitfit:
            raise ValueError(
                'Selective BitFit is mutually exclusive with all_bias_bitfit and '
                'vit_rc_bitfit/vit_gainnorm_bitfit.')

        bitfit_scope = str(getattr(self.pars, 'bitfit_scope', 'all'))
        if bitfit_scope not in ('all', 'no_late3'):
            raise ValueError('Unsupported bitfit_scope: {}'.format(bitfit_scope))
        self.bitfit_scope = bitfit_scope
        if bitfit_scope != 'all' and not standard_bitfit:
            raise ValueError(
                'A non-default bitfit_scope requires standard all_bias_bitfit.')

        descriptors = describe_bias_groups(self.model.named_parameters())
        scale_by_parameter = {
            name: torch.ones_like(parameter.detach())
            for name, parameter in self.model.named_parameters()
            if name.endswith('bias')}

        if headaware_gainnorm_bitfit:
            # The structurally null attention K-bias is excluded from the
            # optimizer from the beginning. Other groups remain at unit scale
            # until the head-aware boundary calibration replaces the buffers.
            for descriptor in descriptors:
                if not descriptor.get('eligible', True):
                    group_view(
                        scale_by_parameter[descriptor['parameter_name']],
                        descriptor).zero_()

        if bitfit_scope == 'no_late3':
            for descriptor in descriptors:
                block = descriptor.get('block')
                excluded = (
                    (block is not None and int(block) >= self.block_depth - 2)
                    or descriptor.get('bias_type') == 'final_norm')
                if excluded:
                    group_view(
                        scale_by_parameter[descriptor['parameter_name']],
                        descriptor).zero_()

        if gainnorm_bitfit:
            manifest_path = str(getattr(
                self.pars, 'bitfit_gain_manifest', '') or '').strip()
            self.bitfit_gain_manifest_path = manifest_path
            with open(manifest_path, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            if (manifest.get('format_version') != 1 or
                    manifest.get('method') != 'frozen_jacobian_gain_norm'):
                raise ValueError('Invalid GainNorm-BitFit manifest format.')
            fingerprint = backbone_bias_fingerprint(
                self.model.named_parameters())
            expected = manifest.get('metadata', {}).get(
                'backbone_bias_sha256')
            if expected is not None and fingerprint != expected:
                raise ValueError(
                    'GainNorm manifest backbone fingerprint differs from the '
                    'loaded pretrained model.')
            rows = {row['group_id']: row for row in manifest['groups']}
            expected_ids = {row['group_id'] for row in descriptors}
            if set(rows) != expected_ids:
                missing = sorted(expected_ids - set(rows))
                extra = sorted(set(rows) - expected_ids)
                raise ValueError(
                    'GainNorm groups do not match model; missing={} extra={}'.format(
                        missing, extra))
            for descriptor in descriptors:
                multiplier = float(rows[descriptor['group_id']]['multiplier'])
                if not math.isfinite(multiplier) or multiplier < 0:
                    raise ValueError(
                        'Invalid GainNorm multiplier for {}.'.format(
                            descriptor['group_id']))
                group_view(
                    scale_by_parameter[descriptor['parameter_name']],
                    descriptor).fill_(multiplier)
            self.bitfit_gain_manifest = manifest
            self._bitfit_calibration_by_group = rows

        # RC-BitFit deliberately trains the exact same bias set as the best
        # standard BitFit baseline. Its controlled change is the layer-wise
        # coding-rate constraint, not a different parameter budget.
        if standard_bitfit:
            for tensor_index, (n, param) in enumerate(
                    self.model.named_parameters()):
                if self._check_params(n, ['bias'], all_match=False):
                    scale = scale_by_parameter[n]
                    selected = scale.ne(0)
                    if not torch.any(selected):
                        continue
                    param.requires_grad = True
                    self.bitfit_optimizer_bias_count += int(param.numel())
                    self.bitfit_effective_bias_count += int(
                        selected.sum().item())
                    if gainnorm_bitfit or headaware_gainnorm_bitfit:
                        scale_name = '_prrd_gain_scale_{}'.format(tensor_index)
                        self.register_buffer(
                            scale_name, scale.detach().clone())
                        self._gain_scale_buffer_by_parameter[n] = scale_name
                        param.register_hook(
                            lambda gradient, buffer_name=scale_name:
                            gradient * getattr(self, buffer_name))
                        if not torch.all(selected):
                            mask_name = '_prrd_gain_mask_{}'.format(tensor_index)
                            initial_name = '_prrd_gain_initial_{}'.format(
                                tensor_index)
                            self.register_buffer(
                                mask_name, selected.detach().clone())
                            self.register_buffer(
                                initial_name, param.detach().clone())
                            self._selective_restore_entries.append((
                                n, param, mask_name, initial_name))
                            self._scr_mask_name_by_bare_name[n] = mask_name
                    if bitfit_lr_scale == 1.0:
                        # Preserve the original optimizer grouping exactly.
                        model_params += [param]
                    else:
                        bitfit_params += [param]
        elif selective_bitfit:
            manifest_path = str(getattr(
                self.pars, 'bitfit_selection_manifest', '') or '').strip()
            self.bitfit_selection_manifest_path = manifest_path
            manifest = load_selection_manifest(manifest_path)
            fingerprint = backbone_bias_fingerprint(
                self.model.named_parameters())
            expected_fingerprint = manifest.get('metadata', {}).get(
                'backbone_bias_sha256')
            if (expected_fingerprint is not None and
                    fingerprint != expected_fingerprint):
                raise ValueError(
                    'Selective BitFit manifest was calibrated on different '
                    'backbone bias values (expected {}, got {}).'.format(
                        expected_fingerprint, fingerprint))
            descriptors, selection_masks = masks_from_manifest(
                self.model.named_parameters(), manifest)
            self.bitfit_selection_manifest = manifest
            self._bitfit_calibration_by_group = {
                row['group_id']: row for row in manifest.get('groups', [])}

            for index, (name, param) in enumerate(
                    self.model.named_parameters()):
                mask = selection_masks.get(name)
                if mask is None:
                    continue
                selected_count = int(torch.count_nonzero(mask).item())
                if selected_count == 0:
                    continue
                param.requires_grad = True
                self.bitfit_optimizer_bias_count += int(param.numel())
                self.bitfit_effective_bias_count += selected_count
                if bitfit_lr_scale == 1.0:
                    model_params += [param]
                else:
                    bitfit_params += [param]

                if selected_count < int(param.numel()):
                    mask_buffer = '_bias_mask_{}'.format(index)
                    initial_buffer = '_bias_initial_{}'.format(index)
                    self.register_buffer(mask_buffer, mask.detach().clone())
                    self.register_buffer(
                        initial_buffer, param.detach().clone())
                    param.register_hook(
                        lambda gradient, buffer_name=mask_buffer:
                        gradient * getattr(self, buffer_name))
                    self._selective_restore_entries.append((
                        name, param, mask_buffer, initial_buffer))
                    self._scr_mask_name_by_bare_name[name] = mask_buffer

            expected_selected = int(manifest['selected_scalar_count'])
            if self.bitfit_effective_bias_count != expected_selected:
                raise ValueError(
                    'Selective BitFit manifest selected {} scalars, but the '
                    'model mask contains {}.'.format(
                        expected_selected, self.bitfit_effective_bias_count))

        if standard_bitfit or selective_bitfit:
            self._bitfit_group_descriptors = descriptors
            self.bitfit_eligible_bias_count = sum(
                int(group['numel']) for group in self._bitfit_group_descriptors
                if group.get('eligible', True))
            self._bitfit_initial_values = {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.model.named_parameters()
                if name.endswith('bias')
            }
            # NOTE: when composed with BEFT, qkv.bias is also managed by the
            # BEFT hook/restore pair.  The snapshot above is taken before the
            # first optimizer step (construction time), so it already holds
            # pristine pretrained values and the two restore passes agree.
            if selective_bitfit:
                _, selection_masks = masks_from_manifest(
                    self.model.named_parameters(),
                    self.bitfit_selection_manifest)
                self._bitfit_selection_masks = {
                    name: mask.detach().cpu().bool()
                    for name, mask in selection_masks.items()
                }
            elif standard_bitfit:
                self._bitfit_selection_masks = {
                    name: scale.ne(0).detach().cpu()
                    for name, scale in scale_by_parameter.items()
                }

        if bias_mask_mode:
            # Stage-D: resolve (targets, include_norm, include_fc2) from the
            # extension table when present; otherwise reproduce the legacy
            # single-target expression exactly.
            _ext = BIAS_MASK_EXTENSIONS.get(self.bias_mask)
            if _ext is not None:
                _extra_targets, include_norm, include_fc2 = _ext
                targets = ('v',) + tuple(_extra_targets)
            else:
                target = 'v' if self.bias_mask == 'v_out' \
                    else self.bias_mask[-1]
                targets = (target,)
                include_norm = self.bias_mask in (
                    'beft_v_norm', 'beft_v_norm_fc2')
                include_fc2 = self.bias_mask == 'beft_v_norm_fc2'
            # Kept for the v_out-only branch further below.
            target = targets[-1]
            # Validate every block before changing requires_grad or optimizer
            # groups so an unsupported architecture fails atomically.
            for block_index, block in enumerate(self.model.blocks):
                audit = audit_attention_qkv_layout(
                    block.attn, self.embed_dim)
                if (self.bias_mask == 'v_out' and
                        block.attn.proj.bias is None):
                    raise RuntimeError(
                        'Block {} attention projection has no bias.'.format(
                            block_index))
                audit['block'] = block_index
                self.beft_layout_audit.append(audit)

            for block_index, block in enumerate(self.model.blocks):
                if block_index not in self.bias_selected_layers:
                    continue
                bias = block.attn.qkv.bias
                dim = self.beft_layout_audit[block_index]['embed_dim']
                # Stage-D: union every requested slice into ONE mask so a
                # multi-target mode (qv = V + q) exposes both
                # slices on the same fused tensor.
                mask = torch.zeros_like(bias, dtype=torch.bool)
                for _slice_name in targets:
                    start, stop = self.beft_layout_audit[block_index][
                        'slices'][_slice_name]
                    mask[start:stop] = True
                # [patched] record the BEFT slice mask (CPU, boolean).
                self._beft_slice_mask_by_name[
                    'blocks.{}.attn.qkv.bias'.format(block_index)] = \
                    mask.detach().cpu().clone()
                mask_name = '_beft_mask_{:03d}'.format(block_index)
                reference_name = '_beft_reference_{:03d}'.format(block_index)
                self.register_buffer(mask_name, mask)
                self.register_buffer(
                    reference_name, bias.detach().clone())
                bias.requires_grad = True
                handle = bias.register_hook(
                    lambda gradient, buffer_name=mask_name:
                    gradient * getattr(self, buffer_name).to(
                        device=gradient.device, dtype=gradient.dtype))
                self._beft_hook_handles.append(handle)
                self._beft_restore_entries.append((
                    'model.blocks.{}.attn.qkv.bias'.format(block_index),
                    bias, mask_name, reference_name))
                self.beft_optimizer_backbone_count += int(bias.numel())
                self.beft_effective_backbone_count += int(mask.sum().item())
                if bitfit_lr_scale == 1.0:
                    model_params.append(bias)
                else:
                    bitfit_params.append(bias)

                if self.bias_mask == 'v_out':
                    proj_bias = block.attn.proj.bias
                    proj_bias.requires_grad = True
                    self.beft_optimizer_backbone_count += int(
                        proj_bias.numel())
                    self.beft_effective_backbone_count += int(
                        proj_bias.numel())
                    if bitfit_lr_scale == 1.0:
                        model_params.append(proj_bias)
                    else:
                        bitfit_params.append(proj_bias)

                # Stage-D extra single group (pure BEFT-V + one group).
                # These are COMPLETE bias tensors; the fused qkv parameter
                # above still exposes only its effective slice.  No forward
                # computation is changed.
                if include_norm:
                    for extra_bias in (block.norm1.bias, block.norm2.bias):
                        if extra_bias is None:
                            raise RuntimeError(
                                'Block {} normalization has no bias.'.format(
                                    block_index))
                        extra_bias.requires_grad = True
                        self.beft_optimizer_backbone_count += int(
                            extra_bias.numel())
                        self.beft_effective_backbone_count += int(
                            extra_bias.numel())
                        if bitfit_lr_scale == 1.0:
                            model_params.append(extra_bias)
                        else:
                            bitfit_params.append(extra_bias)
                if include_fc2:
                    fc2_bias = block.mlp.fc2.bias
                    if fc2_bias is None:
                        raise RuntimeError(
                            'Block {} MLP FC2 has no bias.'.format(block_index))
                    fc2_bias.requires_grad = True
                    self.beft_optimizer_backbone_count += int(
                        fc2_bias.numel())
                    self.beft_effective_backbone_count += int(
                        fc2_bias.numel())
                    if bitfit_lr_scale == 1.0:
                        model_params.append(fc2_bias)
                    else:
                        bitfit_params.append(fc2_bias)

            if include_norm:
                final_norm_bias = self.model.norm.bias
                if final_norm_bias is None:
                    raise RuntimeError('Final backbone norm has no bias.')
                final_norm_bias.requires_grad = True
                self.beft_optimizer_backbone_count += int(
                    final_norm_bias.numel())
                self.beft_effective_backbone_count += int(
                    final_norm_bias.numel())
                if bitfit_lr_scale == 1.0:
                    model_params.append(final_norm_bias)
                else:
                    bitfit_params.append(final_norm_bias)

        # QV_REFINEMENT_V1
        # Static QV (the audited qkv.bias Q/V slices) remains the base model.
        # These optional residuals add either 72 per-head routing scales or a
        # shared low-rank, image-conditioned Q/V bias.  The contextual branch
        # is zero-initialized at its output, so every run starts exactly from
        # the static-QV function rather than a randomly perturbed backbone.
        self.context_rank = int(getattr(
            self.pars, 'context_rank', 0))
        self.context_targets = str(getattr(
            self.pars, 'context_targets', 'qv')).lower()
        self.head_scale_enabled = bool(getattr(
            self.pars, 'head_scale', False))
        if self.context_rank < 0:
            raise ValueError('context_rank must be non-negative.')
        if self.context_targets not in ('q', 'v', 'qv'):
            raise ValueError('context_targets must be q, v, or qv.')
        self.context_refinement_enabled = bool(
            self.context_rank > 0 or self.head_scale_enabled)
        self.context_refinement_parameter_count = 0
        if self.context_refinement_enabled:
            if self.bias_mask != 'qv':
                raise ValueError(
                    'QV refinement requires bias_mask=qv.')
            if self.pars.visual_prompt or self.pars.semantic_proxy:
                raise ValueError(
                    'QV refinement experiments must be pure: VPT/VPTSP off.')
            head_counts = {
                int(block.attn.num_heads) for block in self.model.blocks}
            if len(head_counts) != 1:
                raise RuntimeError(
                    'QV refinement requires a constant attention head count.')
            self.context_num_heads = head_counts.pop()
            refinement_params = []
            if self.context_rank > 0:
                rank = self.context_rank
                self.context_down = nn.Linear(
                    self.embed_dim, rank, bias=False)
                init.xavier_uniform_(self.context_down.weight)
                refinement_params += list(self.context_down.parameters())
                if 'q' in self.context_targets:
                    self.context_up_q = nn.Linear(
                        rank, self.embed_dim, bias=False)
                    init.zeros_(self.context_up_q.weight)
                    self.context_gate_q = nn.Parameter(
                        torch.ones(self.block_depth, rank))
                    refinement_params += list(
                        self.context_up_q.parameters())
                    refinement_params.append(self.context_gate_q)
                if 'v' in self.context_targets:
                    self.context_up_v = nn.Linear(
                        rank, self.embed_dim, bias=False)
                    init.zeros_(self.context_up_v.weight)
                    self.context_gate_v = nn.Parameter(
                        torch.ones(self.block_depth, rank))
                    refinement_params += list(
                        self.context_up_v.parameters())
                    refinement_params.append(self.context_gate_v)
            if self.head_scale_enabled:
                self.qv_log_head_scale = nn.Parameter(torch.zeros(
                    self.block_depth, self.context_num_heads))
                refinement_params.append(self.qv_log_head_scale)
            for parameter in refinement_params:
                parameter.requires_grad = True
            model_params += refinement_params
            self.context_refinement_parameter_count = sum(
                int(parameter.numel()) for parameter in refinement_params)
            # Count these network-owned tensors as backbone adaptation.  Adding
            # the same count to visible/effective keeps the fused-QKV correction
            # in bias_parameter_accounting exact.
            self.beft_optimizer_backbone_count +=                 self.context_refinement_parameter_count
            self.beft_effective_backbone_count +=                 self.context_refinement_parameter_count

        # [patched v3] SCR x BEFT on the SAME tensor: the SCR hook was
        # registered before the BEFT masks existed, so only the SCR slice
        # survived its mask multiplication while the BEFT hook then
        # multiplied the whole gradient by the (disjoint) BEFT slice --
        # zeroing every coordinate.  Widen each SCR mask buffer that shares
        # a tensor with BEFT to the UNION, so both hooks compose.
        if self._selective_restore_entries and self._beft_slice_mask_by_name:
            for entry in self._selective_restore_entries:
                entry_name, entry_param, entry_mask, entry_initial = entry
                beft_extra = self._beft_slice_mask_by_name.get(entry_name)
                if beft_extra is None:
                    continue
                scr_mask = getattr(self, entry_mask)
                beft_extra = beft_extra.to(
                    device=scr_mask.device, dtype=torch.bool)
                if beft_extra.numel() != scr_mask.numel():
                    continue
                widened = (scr_mask.to(dtype=torch.bool) | beft_extra)
                if torch.equal(widened, scr_mask.to(dtype=torch.bool)):
                    continue
                setattr(self, entry_mask, widened.to(dtype=scr_mask.dtype))
                # [patched v3b] the BEFT hook ALSO multiplies the gradient
                # by its own slice mask, and two disjoint 0/1 masks compose
                # to zero.  Widen the BEFT mask buffer to the same union so
                # that both hooks let the Q slice (SCR) and the V slice
                # (BEFT) through.
                for beft_entry in self._beft_restore_entries:
                    beft_name, beft_param, beft_mask_name, _ref = beft_entry
                    bare = (beft_name[6:] if beft_name.startswith('model.')
                            else beft_name)
                    if bare != entry_name:
                        continue
                    beft_buf = getattr(self, beft_mask_name)
                    bunion = (beft_buf.to(dtype=torch.bool)
                              | widened.to(device=beft_buf.device))
                    if not torch.equal(bunion,
                                       beft_buf.to(dtype=torch.bool)):
                        setattr(self, beft_mask_name,
                                bunion.to(dtype=beft_buf.dtype))
                        self._beft_hook_unioned = True
                self._scr_beft_hook_unioned = True
                print('[patched v3] unioned SCR+BEFT masks {} -> {} '
                      'effective scalars'.format(
                          entry_name, int(widened.sum())))

        memory_bank = []
        # self.prompts are the shared task prompts used for sample embeddings;
        # self.p_prompts are indexed by class and feed the semantic-proxy branch.
        if self.pars.visual_prompt:
            self.prompts = self._setup_prompt()
            prompt_params += list(self.prompts.parameters())
            if self.pars.semantic_proxy:
                self.p_prompts = self._setup_prompt(self.n_classes,
                                                self.pars.p_prompt_depth,
                                                self.pars.num_p_prompt,
                                                requires_grad=not self.pars.prompt_memory_bank)

            # depth x n_class_batch x num_p_prompt x embed_dim

            # With many classes the full class-prompt library stays on the CPU and
            # only the classes present in the current batch live in a small trainable
            # GPU buffer; the matching optimiser state is swapped in and out too.
            if self.pars.prompt_memory_bank:
                self.p_prompts_buffer = self._setup_prompt(self.n_classes_batch,
                                                self.pars.p_prompt_depth,
                                                self.pars.num_p_prompt,
                                                init=False)

                ##############################
                # parameters of client will be udpated automatically, but we have to update
                # their optimizing states manually
                # this only works for adamw or sgd, other optimizer may have different states
                ##############################

                self.memory_state_bank = {}
                if self.pars.optim == 'adamw':
                    self.state_key_list = ['step', 'exp_avg', 'exp_avg_sq']
                    for key in self.state_key_list:
                        self.memory_state_bank[key] = []
                        if key == 'step':
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(1))
                        else:
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(self.n_classes,
                                                            self.pars.num_p_prompt,
                                                            self.embed_dim))
                elif self.pars.optim == 'sgd':
                    self.state_key_list = ['momentum_buffer']
                    for key in self.state_key_list:
                        self.memory_state_bank[key] = []
                        if key == 'step':
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(1))
                        else:
                            for i in range(self.pars.p_prompt_depth):
                                self.memory_state_bank[key].append(torch.zeros(self.n_classes,
                                                            self.pars.num_p_prompt,
                                                            self.embed_dim))
                else:
                    raise NotImplementedError
                ##############################

                # save buffer to optmizer
                memory_bank += list(self.p_prompts_buffer.parameters())

            elif self.pars.semantic_proxy:
                proxy_prompt_params += list(self.p_prompts.parameters())

        prompt_lr_scale = float(getattr(
            self.pars, 'prompt_lr_scale', 1.0))
        if prompt_lr_scale < 0:
            raise ValueError('prompt_lr_scale must be non-negative.')
        prompt_lr = opt.lr * prompt_lr_scale

        # setup optim_dict_list
        self.optim_bank_idx = 1
        self.optim_dict_list = [
            {
                'params': nn.ParameterList(proxy_net_params),
                'lr': opt.lr,
                'weight_decay': opt.decay
            },
                 #optim_bank_idx = 1
            {
                'params': nn.ParameterList(memory_bank),
                'lr': prompt_lr,
                'weight_decay': opt.decay
            },
            {
                'params': nn.ParameterList(model_params),
                'lr': opt.lr,
                'weight_decay': opt.decay
            }]
        if prompt_params:
            self.optim_dict_list.append({
                'params': nn.ParameterList(prompt_params),
                'lr': prompt_lr,
                'weight_decay': opt.decay
            })
        if proxy_prompt_params:
            self.optim_dict_list.append({
                'params': nn.ParameterList(proxy_prompt_params),
                'lr': prompt_lr,
                'weight_decay': opt.decay
            })
        if bitfit_params:
            self.optim_dict_list.append({
                'params': nn.ParameterList(bitfit_params),
                'lr': opt.lr * bitfit_lr_scale,
                'weight_decay': opt.decay
            })

        self.p_optim_dict_list = [{
                'params': nn.ParameterList([self.proxies_bias]),
                'lr': opt.lr_proxy,
                'weight_decay': opt.decay
            }]
        self.model.cuda().train()

    def _build_backbone(self, opt):
        """Build the timm ViT wrapped by the adaptation code.
        Subclasses may override this to restrict the backbone family."""
        # Passing img_size lets timm resample pretrained absolute positional
        # embeddings (e.g. DINOv2 518 -> 224) during checkpoint loading.
        model_kwargs = {'img_size': opt.model_input_size}
        checkpoint_path = str(
            getattr(opt, 'pretrained_checkpoint', '') or '').strip()
        checkpoint_path = os.path.abspath(os.path.expanduser(
            checkpoint_path)) if checkpoint_path else ''
        if checkpoint_path and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'Local ViT pretrained checkpoint does not exist: {}'.format(
                    checkpoint_path))
        if 'dinov2' in opt.pretrained_model.lower():
            # timm also registers Meta's official DINOv2 URL. Prefer it here
            # so this path does not depend on the Hugging Face CDN.
            model_kwargs['pretrained_cfg_overlay'] = {'hf_hub_id': None}
        try:
            return timm.create_model(
                opt.pretrained_model,
                pretrained=not checkpoint_path,
                checkpoint_path=checkpoint_path or None,
                **model_kwargs)
        except Exception as error:
            if checkpoint_path:
                raise RuntimeError(
                    'Failed to load local ViT pretrained checkpoint: {}'.format(
                        checkpoint_path)) from error
            raise RuntimeError(
                'ViT pretrained weights could not be downloaded. Provide '
                '--pretrained_checkpoint or set DML_PRETRAINED_CHECKPOINT to '
                'a local timm-format model.safetensors file.') from error

    # save states to bank
    def bank_optim_states_save(self):
        with torch.no_grad():
            for param_idx, param in enumerate(
                    self.optimizer.param_groups[self.optim_bank_idx]['params']
                ):
                state = self.optimizer.state[param]
                
                if len(state.keys()) == 0:
                    continue
                
                for k in self.state_key_list:
                    # n_classes_batch x num_p_prompt x embed_dim should be same to the buffer size
                    if k == 'step':
                        self.memory_state_bank[k][param_idx] = state[k].cpu()
                    else:
                        self.memory_state_bank[k][param_idx][self.u_labels_local] = state[k].cpu()
    
    # save model to bank
    def bank_prompt_p_save(self):
        with torch.no_grad():
            for idx, prompt in enumerate(self.p_prompts_buffer):
                prompt_local = prompt.detach().cpu()
                prompt_local.requires_grad = False
                self.p_prompts[idx][self.u_labels_local] = prompt_local
    
    def _update_optim_parameters(self, p_depth_idx, update_values, update_states):
        with torch.no_grad():
            param = self.optimizer.param_groups[self.optim_bank_idx]['params']
            
            # Update parameter values
            param[p_depth_idx].copy_(update_values)
            
            # Update optimizer state (e.g., momentum)
            if len(self.optimizer.state.keys()) > 0 and \
                len(self.optimizer.state[param[p_depth_idx]]) > 0:

                for k in self.state_key_list:
                    self.optimizer.state[param[p_depth_idx]][k].copy_(update_states[k])
    
    # load model and state from bank
    def bank_prompt_p_widthdraw(self):
        with torch.no_grad():
            for idx, prompt in enumerate(self.p_prompts):
                _prompt_local = prompt[self.u_labels_local].clone().to(self.pars.device) # BS x num x 384
                _prompt_local.requires_grad = True
                update_state = {}
                for key in self.state_key_list:
                    if key == 'step':
                        update_state[key] = self.memory_state_bank[key][idx].clone().to(self.pars.device)
                    else:
                        update_state[key] = self.memory_state_bank[key][idx][self.u_labels_local].clone().to(self.pars.device)
                # update parameters and states to optimizer
                self._update_optim_parameters(idx, _prompt_local, update_state)
    
    
    def _recover_indices(self, input_tensor, target_tensor):
        with torch.no_grad():
            idx = torch.nonzero(torch.eq(input_tensor.view(-1, 1), target_tensor).T, as_tuple=True)[1]
        return idx
    
    # check params name
    def _check_params(self, module_name, safe_list, all_match=True):
        check = [partial_name in module_name for partial_name in safe_list]
        return all(check) if all_match else any(check)
    
    def _cal_channels(self):
        conv = self.model.patch_embed.proj
        in_height = self.input_size[1]
        in_width = self.input_size[2]
        out_height = (in_height + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        out_width = (in_width + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
        return out_height * out_width + self._num_prefix_tokens()

    def _num_prefix_tokens(self):
        """Return the number of non-spatial tokens emitted by the backbone."""
        return int(getattr(self.model, 'num_prefix_tokens', 1))

    def _spatial_tokens(self, x):
        """Drop CLS/register/distillation tokens before spatial pooling."""
        return x[:, self._num_prefix_tokens():]
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        else:
            raise NotImplementedError("Not init implemented for {}".format(module))
    
    def _setup_head(self, in_dim=None, out_dim=None):
        if in_dim is None:
            in_dim = self.model.embed_dim
        if out_dim is None:
            out_dim = self.pars.embed_dim
        
        head = nn.Linear(in_dim, out_dim).apply(self._init_weights)
        norm = nn.LayerNorm(in_dim).apply(self._init_weights)
        for _, param in head.named_parameters():
            param.requires_grad = True
        for _, param in norm.named_parameters():
            param.requires_grad = True
        
        if self.distill:
            dist_head = nn.Linear(in_dim, out_dim).apply(self._init_weights)
            dist_norm = nn.LayerNorm(in_dim).apply(self._init_weights)  
            for _, param in dist_head.named_parameters():
                param.requires_grad = True
            for _, param in dist_norm.named_parameters():
                param.requires_grad = True
            return head, norm, dist_head, dist_norm
        
        return head, norm, None, None
    
    def _copy_parameters(self, A: nn.Module, B: nn.Module):
        dict_A = A.state_dict()
        dict_B = B.state_dict()

        new_dict_B = dict_B.copy()
        for name_A, param_A in dict_A.items():
            if name_A in dict_B:
                if dict_B[name_A].shape == param_A.shape:
                    new_dict_B[name_A] = param_A
        B.load_state_dict(new_dict_B)
    
    def _setup_adapter(self, model_params):
        depth = self.block_depth
        adapter_num_blocks = max(min(self.pars.adapter_num_blocks, depth), 0)
        for i in range(depth-1, depth - adapter_num_blocks-1, -1):
            adapter = AdaptBlock(self.pars, 
                                dim=self.model.embed_dim,
                                drop_path=self.pars.feature_dropout)
            
            self._copy_parameters(self.model.blocks[i], adapter)
            self.model.blocks[i] = adapter
            
            for _, param in self.model.blocks[i].named_parameters():
                param.requires_grad = False
            for _, param in self.model.blocks[i].adapter.named_parameters():
                param.requires_grad = True
                model_params += [param]
        return model_params
    
    def _setup_prompt(self, length=None, depth=None, num_prompt=None, requires_grad=True, init=True):
        if depth is None:
            depth = max(min(self.pars.prompt_depth, self.block_depth), 0)
            self.pars.prompt_depth = depth
        if num_prompt is None:
            num_prompt = self.pars.num_prompt
        if length is None:
            length = 1
        # Setup VPTs
        # depth x n_class x num_prompt x embed_dim
        if init:
            init_scale = float(getattr(
                self.pars, 'prompt_init_scale', 1.0))
            if init_scale < 0:
                raise ValueError('prompt_init_scale must be non-negative.')
            prompts = nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(
                    torch.zeros(length, num_prompt, self.embed_dim))
                    * init_scale)
                 for _ in range(depth)])
        else:
            prompts = nn.ParameterList(
                [nn.Parameter(
                    torch.zeros(length, num_prompt, self.embed_dim))
                 for _ in range(depth)])
        
        if requires_grad:
            for _, param in prompts.named_parameters():
                param.requires_grad = True
        else:
            for _, param in prompts.named_parameters():
                param.requires_grad = False
        return prompts
    
    def _setup_proxies(self):
        '''setup proxy prompts'''
        proxies = nn.Parameter(nn.init.normal_(
            torch.empty(self.n_classes, self.p_channels, self.embed_dim),
                        mean=0.5, std=0.5)
                                )
        proxies.requires_grad = True
        return proxies
    
    
    def _print_trainable_params(self):
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n + ' ' + str(p.shape) + '\n')

    def enforce_bias_slice_mask(self):
        """Restore every frozen slice after optimizer weight decay/momentum.

        A gradient hook alone is insufficient because SGD/AdamW can still
        modify zero-gradient coordinates through weight decay.  Restoring the
        pristine values makes the virtual Q/K/V and group masks exact.

        When composed with a BEFT mode, the BEFT restore pass has already run
        and kept only the BEFT slice (e.g. the V slice of qkv.bias).  This pass
        runs SECOND, so the mask used here must also include the BEFT slice --
        otherwise it reverts the BEFT slice to pristine on every step and the
        BEFT contribution silently vanishes.  We therefore widen the effective
        mask to the union of the SCR group and the BEFT slice.
        """
        if not self._selective_restore_entries:
            return
        with torch.no_grad():
            for name, parameter, mask_name, initial_name in \
                    self._selective_restore_entries:
                mask = getattr(self, mask_name).to(dtype=torch.bool)
                slice_masks = getattr(self, '_beft_slice_mask_by_name', None)
                if slice_masks is not None:
                    extra = slice_masks.get(name)
                    if extra is not None:
                        extra = extra.to(device=mask.device, dtype=torch.bool)
                        if extra.numel() == mask.numel():
                            mask = mask | extra
                initial = getattr(self, initial_name).to(
                    device=parameter.device, dtype=parameter.dtype)
                mask = mask.to(device=parameter.device)
                parameter.copy_(torch.where(mask, parameter, initial))

    def restore_frozen_bias_slices(self):
        """Undo optimizer decay/momentum updates outside the BEFT slice."""
        if not self._beft_restore_entries:
            return
        with torch.no_grad():
            for name, parameter, mask_name, reference_name in \
                    self._beft_restore_entries:
                bare_name = (name[6:] if name.startswith('model.')
                             else name)
                mask = getattr(self, mask_name).to(
                    device=parameter.device, dtype=torch.bool)
                # [patched] composed runs: also preserve the SCR
                # group, so this pass does not destroy it before
                # the SCR pass gets a chance to run.
                scr_lookup = getattr(
                    self, '_scr_mask_name_by_bare_name', None)
                if scr_lookup:
                    scr_name = scr_lookup.get(bare_name)
                    if scr_name is not None:
                        scr_mask = getattr(self, scr_name, None)
                        if scr_mask is not None:
                            scr_mask = scr_mask.to(
                                device=mask.device, dtype=torch.bool)
                            if scr_mask.numel() == mask.numel():
                                mask = mask | scr_mask
                reference = getattr(self, reference_name).to(
                    device=parameter.device, dtype=parameter.dtype)
                parameter.copy_(torch.where(mask, parameter, reference))

    def bias_parameter_accounting(self):
        """Return optimizer-visible and slice-effective trainable counts."""
        visible_total = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad)
        effective_total = (
            visible_total - self.beft_optimizer_backbone_count +
            self.beft_effective_backbone_count)
        return {
            'optimizer_visible_trainable_params': int(visible_total),
            'effective_trainable_params': int(effective_total),
            'backbone_optimizer_visible_trainable_params': int(
                self.beft_optimizer_backbone_count),
            'backbone_effective_trainable_params': int(
                self.beft_effective_backbone_count),
        }

    def apply_headaware_gain_manifest(self, manifest, manifest_path):
        """Install boundary-calibrated multipliers into existing hooks."""
        if not getattr(self.pars, 'vit_headaware_gainnorm_bitfit', False):
            raise RuntimeError(
                'A head-aware manifest can only be applied in its BitFit mode.')
        if (manifest.get('format_version') != 1 or
                manifest.get('method') !=
                'head_aware_embedding_jacobian_gain_norm'):
            raise ValueError('Invalid Head-aware GainNorm manifest format.')
        rows = {row['group_id']: row for row in manifest.get('groups', [])}
        expected_ids = {
            descriptor['group_id']
            for descriptor in self._bitfit_group_descriptors}
        if set(rows) != expected_ids:
            raise ValueError(
                'Head-aware GainNorm groups do not match the model.')
        named_parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for buffer_name in self._gain_scale_buffer_by_parameter.values():
                getattr(self, buffer_name).zero_()
            for descriptor in self._bitfit_group_descriptors:
                row = rows[descriptor['group_id']]
                multiplier = float(row['multiplier'])
                if not math.isfinite(multiplier) or multiplier < 0:
                    raise ValueError(
                        'Invalid multiplier for {}.'.format(
                            descriptor['group_id']))
                parameter_name = descriptor['parameter_name']
                if parameter_name not in named_parameters:
                    raise ValueError(
                        'Unknown calibrated parameter: {}'.format(
                            parameter_name))
                buffer_name = self._gain_scale_buffer_by_parameter.get(
                    parameter_name)
                if buffer_name is None:
                    if multiplier != 0:
                        raise ValueError(
                            'Calibrated nonzero group lacks a gradient hook: '
                            '{}'.format(descriptor['group_id']))
                    continue
                group_view(
                    getattr(self, buffer_name), descriptor).fill_(multiplier)
        self.bitfit_gain_manifest = manifest
        self.bitfit_gain_manifest_path = str(manifest_path)
        self._bitfit_calibration_by_group = rows

    def bias_diagnostic_payload(self, epoch=None):
        """Return exact per-group update norms for JSONL experiment records."""
        if not self._bitfit_group_descriptors:
            return None
        current_parameters = dict(self.model.named_parameters())
        rows = []
        for descriptor in self._bitfit_group_descriptors:
            name = descriptor['parameter_name']
            current = current_parameters[name].detach().cpu().float()
            initial = self._bitfit_initial_values[name].float()
            delta = group_view(current - initial, descriptor)
            mask = group_view(
                self._bitfit_selection_masks[name], descriptor)
            calibration = self._bitfit_calibration_by_group.get(
                descriptor['group_id'], {})
            row = dict(descriptor)
            row.update({
                'selected_scalar_count': int(mask.sum().item()),
                'selected_fraction': float(mask.float().mean().item()),
                'delta_l2': float(delta.norm().item()),
                'delta_rms': float(delta.square().mean().sqrt().item()),
                'delta_max_abs': float(delta.abs().max().item()),
                'calibration_score': calibration.get('score'),
                'calibration_cv': calibration.get('cv'),
                'gain_multiplier': calibration.get('multiplier'),
            })
            rows.append(row)
        if self.bitfit_selection_manifest is not None:
            objective = self.bitfit_selection_manifest.get('objective')
        elif self.bitfit_gain_manifest is not None:
            objective = self.bitfit_gain_manifest.get('method')
        elif self.bitfit_scope != 'all':
            objective = 'all_bias_' + self.bitfit_scope
        else:
            objective = 'all_bias'
        return {
            'epoch': None if epoch is None else int(epoch),
            'selection_manifest': (
                self.bitfit_selection_manifest_path or
                self.bitfit_gain_manifest_path or None),
            'selection_objective': objective,
            'bitfit_scope': self.bitfit_scope,
            'optimizer_backbone_bias_scalars': int(
                self.bitfit_optimizer_bias_count),
            'effective_selected_bias_scalars': int(
                self.bitfit_effective_bias_count),
            'eligible_non_key_bias_scalars': int(
                self.bitfit_eligible_bias_count),
            'groups': rows,
        }
                
    def _split_tensor(self, A, k, stack=True):
        # return k x (n//k) x dim
        assert (A.shape[0] % k == 0) & (k > 0)
        split_tensors = [A[i::k] for i in range(k)] # list of k x (BS//k) x dim
        if stack:
            return torch.stack(split_tensors)
        else:
            return split_tensors
        
    def forward_head(self, x, **kwargs):
        # new head for dml
        if self.pars.pooling == 'avg':
            x = self._spatial_tokens(x).mean(dim=1)  # avg pooling
        elif self.pars.pooling == 'combined':
            x = torch.cat((x[:, 0], self._spatial_tokens(x).mean(dim=1)),
                          dim=-1)
        else:
            if self.distill:
                x_dist = x[:, 1]
            x = x[:, 0]  # use cls_token
        
        # train head or run zero-shot
        if self.pars.train_head:
            x = self.fc_norm(x)
            x = self.head(x)
            if self.distill:
                x_dist = self.dist_norm(x_dist)
                x_dist = self.dist_head(x_dist)
                x = (x + x_dist) / 2
        
        # Hyp-Metric keeps tangent features un-normalized, then maps them to
        # the Poincare ball (c>0); c=0 is the exact spherical baseline.
        x = self.hyp_projection(x) if self.hyp_metric_enabled else F.normalize(x, dim=-1)
        return x
    
    
    def forward_p_head(self, x, **kwargs):
        if self.pars.pooling == 'avg':
            x = self._spatial_tokens(x).mean(dim=1)  # avg pooling
        elif self.pars.pooling == 'combined':
            x = torch.cat((x[:, 0], self._spatial_tokens(x).mean(dim=1)),
                          dim=-1)  # cls+avg 2 x embed_dim
        else:
            if self.distill:
                x_dist = x[:, 1]
            x = x[:, 0]  # use cls_token
        
        # train head or run zero-shot
        if self.pars.train_head:
            x = self.p_fc_norm(x)
            x = self.p_head(x)
            if self.distill:
                x_dist = self.p_dist_norm(x_dist)
                x_dist = self.p_dist_head(x_dist)
                x = (x + x_dist) / 2
            
        x = F.normalize(x, dim=-1)
        return x
    
    
    def forward_proxy_mix_bias(self, x, labels, **kwargs):
        # The classic proxy looks up a free vector per class. Here the same images
        # go through a second frozen ViT path and a class prompt is selected by
        # label, so the candidate proxies carry visual semantics from the input.
        
        f_labels = labels.view(-1)
        # unique labels in each batch: BS//k

        self.f_labels = labels.view(-1)
        self.u_labels = torch.unique_consecutive(self.f_labels) # BS//k, in order
        self.BS = self.f_labels.shape[0]
        self.k = self.pars.samples_per_class
        self.u_labels_local = self.u_labels.cpu() # BS//k, in order
        
        assert x.shape[0] == f_labels.shape[0]
        
        batch_p_prompts = []
        
        if self.pars.prompt_memory_bank:
            self.batch_memory_bank = []
            self.bank_prompt_p_widthdraw()

        ## ================== memory bank ==================
        for idx in range(self.pars.p_prompt_depth):
            # assign prompt frmo bank to buffer

            if not self.pars.prompt_memory_bank:
                prompt = self.p_prompts[idx][f_labels] # BS x num x 384
            else:
                r_idx = self._recover_indices(self.u_labels, f_labels)
                prompt = self.p_prompts_buffer[idx][r_idx]
            
            batch_p_prompts.append(prompt)
        
        
        ## ================== pre_embedding ==================
        x, backbone_context = self._pre_embed(x)
        
        p = x
        
        ## ==================== semantic-proxy branch ====================
        for idx, blk in enumerate(self.model.blocks):
            # transfer i to block index
            if self.pars.visual_prompt:
                # add vpt with the running blocks
                num_p_vpt = 0
                num_vpt = 0
                vpt = torch.tensor([]).to(x.device)
                if idx < self.pars.p_prompt_depth:
                    num_p_vpt = max((self.pars.num_p_prompt - idx * self.pars.prompt_p_block_step), 0)
                    if num_p_vpt > 0:
                        p_prompts = batch_p_prompts[idx][:, :num_p_vpt, :]
                        vpt = torch.cat([p_prompts, vpt], dim=1)
                
                p = self._forward_vpt_block(
                    p, blk, vpt, num_vpt + num_p_vpt,
                    block_index=idx, backbone_context=backbone_context)
            else:
                p = self._forward_backbone_block(
                    p, blk, block_index=idx,
                    backbone_context=backbone_context)
        p = self.model.norm(p) # B x 197 x 384
        
        return self._proxies_generator(p)
    
    # Accumulate the per-image candidate proxies into one persistent semantic
    # representation P_c per class.
    def _semantic_mix_with_container(self, p):
        pc = self.proxies_container[self.u_labels] # BS//k x 384
        if self.pars.semantic_mix_type == 'mix':
            # EMA-style accumulation: average the same-class candidates of the
            # current batch and blend them into the class history with lambda.
            p = self._split_tensor(p, self.k).mean(dim=0) # BS//k x 384
            pc = self.pars.proxy_mix_keep_ratio * pc + (1 - self.pars.proxy_mix_keep_ratio) * p
        elif 'gru' in self.pars.semantic_mix_type:
            # GRU-style accumulation: P_c is the hidden state and receives the
            # same-class candidates one by one. Gradients do not cross batches.
            p = p.view(self.BS//self.k, self.k, -1)
            h = self.proxies_container[self.u_labels]
            for i in range(self.k):
                h = self.rnn_update(p[:, i, :], h)
            pc = h
        else:
            raise NotImplementedError
        self.proxies_container[self.u_labels] = pc
        self.proxies_container = F.normalize(self.proxies_container, dim=-1)
        return pc
    
    def _update_proxy_bias(self, p):
        with torch.no_grad():
            self.proxies_bias[self.u_labels].copy_(p)
    
    
    def _proxies_generator(self, p):
        
        if self.pars.pooling == 'avg':
            p = self._spatial_tokens(p).mean(dim=1)  # avg pooling
        elif self.pars.pooling == 'combined':
            p = torch.cat((p[:, 0], self._spatial_tokens(p).mean(dim=1)),
                          dim=-1)
        else:
            if self.distill:
                p_dist = p[:, 1]
            p = p[:, 0] # B x 384

        # train head, probing or run zero-shot
        if self.pars.train_head:
            p = self.p_fc_norm(p)
            p = self.p_head(p)
            
            if self.distill:
                p_dist = self.p_dist_norm(p_dist)
                p_dist = self.p_dist_head(p_dist)
                p = (p + p_dist) / 2
            
        p = F.normalize(p, dim=-1)
        
        p = self._semantic_mix_with_container(p)
        
        # Fusion Q_c = alpha * O_c + (1 - alpha) * P_c. This is the key output
        # change with respect to the classic proxy-anchor objective: the loss
        # receives proxies carrying data semantics, not only the free O_c.
        p = self.proxies_bias[self.u_labels] * self.pars.proxy_bias_ratio + \
            p * (1 - self.pars.proxy_bias_ratio)
        
        p_all = self.proxies_bias * self.pars.proxy_bias_ratio + \
            self.proxies_container * (1 - self.pars.proxy_bias_ratio)
        p = F.normalize(p, dim=-1)
        
        # Gradients are deliberately not propagated across iterations or batches.
        self.proxies_container = self.proxies_container.detach()
        
        return p_all, p
    
    
    def forward_simple_proxies(self, x, labels, **kwargs):
        
        self.f_labels = labels.view(-1)
        self.u_labels = torch.unique_consecutive(self.f_labels) # BS//k, in order
        self.BS = self.f_labels.shape[0]
        self.k = self.pars.samples_per_class
        
        x, backbone_context = self._pre_embed(x)
        
        if not self.pars.visual_prompt:
            for idx, blk in enumerate(self.model.blocks):
                x = self._forward_backbone_block(
                    x, blk, block_index=idx,
                    backbone_context=backbone_context)
        else:
            for idx, blk in enumerate(self.model.blocks):
                if idx < self.pars.prompt_depth and self.pars.simple_proxies_vpt:
                    num_vpt = max((self.pars.num_prompt - idx * self.pars.prompt_block_step), 0)
                    prompts = self.prompts[idx][:, :num_vpt, :]
                    vpt = prompts.repeat(x.shape[0], 1, 1) # 1 x N x 384 -> B x N x 384
                    x = self._forward_vpt_block(
                        x, blk, vpt, num_vpt, block_index=idx,
                        backbone_context=backbone_context)
                else:
                    x = self._forward_backbone_block(
                        x, blk, block_index=idx,
                        backbone_context=backbone_context)
        p = self.model.norm(x)
        
        return self._proxies_generator(p)
    
    
    def _forward_contextual_attention(self, x, attention,
                                      block_index):
        """Attention with static QV plus optional contextual/scaling terms."""
        batch, tokens, channels = x.shape
        qkv = attention.qkv(x).reshape(
            batch, tokens, 3, attention.num_heads,
            attention.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.context_rank > 0:
            # x is already Block.norm1(x).  Excluding CLS makes z depend on the
            # visual token set and not on the learned/global CLS carrier.
            spatial = x[:, 1:, :] if tokens > 1 else x
            z = self.context_down(spatial.mean(dim=1))
            if 'q' in self.context_targets:
                dynamic_q = self.context_up_q(
                    z * self.context_gate_q[block_index])
                q = q + dynamic_q.reshape(
                    batch, attention.num_heads, 1, attention.head_dim)
            if 'v' in self.context_targets:
                dynamic_v = self.context_up_v(
                    z * self.context_gate_v[block_index])
                v = v + dynamic_v.reshape(
                    batch, attention.num_heads, 1, attention.head_dim)

        q, k = attention.q_norm(q), attention.k_norm(k)
        q = q * attention.scale
        logits = q @ k.transpose(-2, -1)
        if self.head_scale_enabled:
            alpha = self.qv_log_head_scale[block_index].exp().reshape(
                1, attention.num_heads, 1, 1)
            logits = logits * alpha
        weights = attention.attn_drop(logits.softmax(dim=-1))
        output = (weights @ v).transpose(1, 2).reshape(
            batch, tokens, channels)
        output = attention.proj(output)
        return attention.proj_drop(output)

    def _run_backbone_block(self, x, unit, block_index=None,
                            backbone_context=None, num_vpt=0):
        """Run one transformer block; specialized backbones may add context."""
        if not self.context_refinement_enabled:
            return unit(x)
        if block_index is None:
            raise ValueError('QV refinement requires a transformer block index.')
        attention_output = self._forward_contextual_attention(
            unit.norm1(x), unit.attn, int(block_index))
        x = x + unit.drop_path1(unit.ls1(attention_output))
        x = x + unit.drop_path2(unit.ls2(unit.mlp(unit.norm2(x))))
        return x

    def _forward_backbone_block(self, x, unit, block_index=None,
                                backbone_context=None, num_vpt=0):
        def block_forward(tokens):
            return self._run_backbone_block(
                tokens, unit, block_index=block_index,
                backbone_context=backbone_context, num_vpt=num_vpt)

        if not self.pars.checkpoint:
            return block_forward(x)
        return checkpoint(block_forward, x)

    def _forward_vpt_block(self, x, unit, vpt, num_vpt, block_index=None,
                           backbone_context=None):
        s = x.shape[1]
        if num_vpt > 0:
            x = torch.cat([vpt, x], dim=1)
        x = self._forward_backbone_block(
            x, unit, block_index=block_index,
            backbone_context=backbone_context, num_vpt=num_vpt)
        x = x[:, num_vpt:, :]
        assert x.shape[1] == s
        return x

    def _pre_embed(self, x):
        """Create input tokens and optional per-block positional context."""
        if self.distill:
            return self._dist_pre_embed(x), None
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.norm_pre(x)
        return x, None
    
    def _dist_pre_embed(self, x):
        assert self.distill
        x = self.model.patch_embed(x)
        x = torch.cat((
            self.model.cls_token.expand(x.shape[0], -1, -1),
            self.model.dist_token.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        return x
    
    def forward_feature(self, x, collect_rc_features=False, **kwargs):
        
        x, backbone_context = self._pre_embed(x)
        rc_features = []
        if collect_rc_features:
            start_layer = int(self.pars.rc_bitfit_start_layer)
            if start_layer < 1 or start_layer > self.block_depth:
                raise ValueError(
                    'rc_bitfit_start_layer must be in [1, {}], got {}.'.format(
                        self.block_depth, start_layer))
            if start_layer == 1:
                # Patch-embedded CLS is the reference for transformer layer 1.
                rc_features.append(x[:, 0])
        
        if not self.pars.visual_prompt:
            for idx, blk in enumerate(self.model.blocks):
                x = self._forward_backbone_block(
                    x, blk, block_index=idx,
                    backbone_context=backbone_context)
                if (collect_rc_features and
                        idx + 1 >= max(1, start_layer - 1)):
                    rc_features.append(x[:, 0])
        else:
            for idx, blk in enumerate(self.model.blocks):
                if idx < self.pars.prompt_depth:
                    num_vpt = max((self.pars.num_prompt - idx * self.pars.prompt_block_step), 0)
                    prompts = self.prompts[idx][:, :num_vpt, :]
                    vpt = prompts.repeat(x.shape[0], 1, 1) # 1 x N x 384 -> B x N x 384
                    x = self._forward_vpt_block(
                        x, blk, vpt, num_vpt, block_index=idx,
                        backbone_context=backbone_context)
                else:
                    x = self._forward_backbone_block(
                        x, blk, block_index=idx,
                        backbone_context=backbone_context)
                if (collect_rc_features and
                        idx + 1 >= max(1, start_layer - 1)):
                    rc_features.append(x[:, 0])
        x = self.model.norm(x)
        if collect_rc_features:
            return x, tuple(rc_features)
        return x

    def _reference_backbone_cls(self, x):
        """Run the pristine teacher on the exact same augmented images."""
        if not self.reference_geometry_enabled:
            return None
        if self.pars.visual_prompt or self.distill:
            raise ValueError(
                'PRRD-v1 reference geometry supports the no-prompt, '
                'non-distilled DINO path only.')
        # model.train() recursively toggles registered children, so restore
        # the teacher's deterministic mode on every training forward.
        self.reference_model.eval()
        with torch.no_grad():
            reference = self.reference_model.forward_features(x)
        if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
            raise TypeError(
                'The frozen reference backbone must return [B, tokens, D].')
        return reference[:, 0].detach()

    def _head_from_cls(self, cls_features):
        """Map CLS features through the shared DML head without backbone flow."""
        if not self.pars.train_head:
            raise ValueError('Head relation calibration requires train_head.')
        if self.pars.pooling != 'token' or self.distill:
            raise ValueError(
                'Head relation calibration currently requires token pooling '
                'and a non-distilled backbone.')
        return F.normalize(
            self.head(self.fc_norm(cls_features)), dim=-1)
    
    def forward(self, x, labels=None, anchor=None, warmup=False,
                compute_proxies=True, **kwargs):
        x_res = x

        # [NIR-style warmup]
        # For the first warmup epochs the sample-branch feature extractor (ViT,
        # shared prompts, bias slices) is detached so that the DML projection
        # head adapts to the task first. The semantic-proxy branch runs separately
        # below and still trains the class prompts, its own head and the
        # accumulator; proxies_bias keeps updating as usual. The full gradient
        # path is restored once warmup ends.
        feature_context = torch.no_grad() if warmup else contextlib.nullcontext()
        collect_rc_features = (
            getattr(self.pars, 'vit_rc_bitfit', False) and self.training and
            labels is not None)
        with feature_context:
            feature_output = self.forward_feature(
                x, collect_rc_features=collect_rc_features, **kwargs)
        if collect_rc_features:
            x, rc_features = feature_output
        else:
            x, rc_features = feature_output, None
        backbone_features = x[:, 0]
        reference_backbone_features = None
        head_relation_embeddings = None
        reference_head_embeddings = None
        if (self.reference_geometry_enabled and self.training and
                labels is not None and
                (not warmup or self.rate_projection_enabled)):
            reference_backbone_features = self._reference_backbone_cls(x_res)
            if (self.pars.loss == 'head_relation_budget_proxy_peft' and
                    not warmup):
                # The auxiliary sees the current backbone only as a detached
                # input. Its gradient therefore updates fc_norm/head, never
                # the backbone; PA remains the sole task gradient to BitFit.
                head_relation_embeddings = self._head_from_cls(
                    backbone_features.detach())
                with torch.no_grad():
                    reference_head_embeddings = self._head_from_cls(
                        reference_backbone_features).detach()
        x = self.forward_head(x, **kwargs)
        x = self.dropout(x)
        
        # The semantic-proxy path needs images and labels to update the class
        # state, so it is enabled during training only; when disabled the model
        # falls back to the classic learnable proxies_bias.
        if self.pars.semantic_proxy and self.training and compute_proxies:
            if self.pars.proxy_mix_bias:
                pc = self.forward_proxy_mix_bias(x_res, labels, **kwargs)
            elif self.pars.simple_proxies:
                pc = self.forward_simple_proxies(x_res, labels, **kwargs)
            else:
                raise NotImplementedError ("Not implemented proxy net")
        else:
            if labels is not None and compute_proxies:
                self.u_labels = torch.unique_consecutive(labels)
                pc = (self.proxies_bias, self.proxies_bias[self.u_labels])
            else:
                pc = None
        return {
            'embeds': x,
            'proxies': pc,
            # The stable classic proxy O_c conditions the auxiliary flow objective
            # while the fused semantic proxy Q_c still serves the proxy-anchor loss,
            # so accumulator drift cannot disturb the flow.
            'nir_proxies': self.proxies_bias,
            'rc_features': rc_features,
            'backbone_features': backbone_features,
            'reference_backbone_features': reference_backbone_features,
            'head_relation_embeddings': head_relation_embeddings,
            'reference_head_embeddings': reference_head_embeddings,
            'labels': labels
        }

def init_bert_weights(module):
    """Initialize the weights for BERT."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
        

class Simple_GRUCell(nn.Module):
    # Semantic proxy accumulator. The candidate-state nonlinearity is ReLU
    # instead of the standard GRU tanh (semantic_mix_type="gru_relu") so that
    # vector scale is preserved; the hidden state is the previous class
    # semantic proxy rather than a generic sequence state.
    def __init__(self, input_size, hidden_size, drop=0.0, type='gru'):
        super(Simple_GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_drop = nn.Dropout(drop)
        self.type = type
        self.W_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_ir = nn.Parameter(torch.Tensor(hidden_size))

        self.W_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_iz = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_in = nn.Parameter(torch.Tensor(hidden_size))
        
        self.W_ia = nn.Parameter(torch.Tensor(input_size, 1))
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
    
    def forward(self, x, h_prev):
        
        z = torch.sigmoid(x @ self.W_iz + h_prev @ self.W_hz + self.b_iz)
        r = torch.sigmoid(x @ self.W_ir + h_prev @ self.W_hr + self.b_ir)
        if self.type == 'gru':
            n = torch.tanh(x @ self.W_in + r * (h_prev @ self.W_hn) + self.b_in)
            h_next = (1 - z) * n + z * h_prev
        elif self.type == 'gru_relu':
            # Variant used here: the candidate state uses ReLU, not tanh.
            n = F.relu(x @ self.W_in + r * (h_prev @ self.W_hn) + self.b_in)
            h_next = (1 - z) * n + z * h_prev
        else:
            raise NotImplementedError
        
        h_next = self.rnn_drop(h_next)
        return h_next




# https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py
class AdapterLayer(nn.Module):
    def __init__(self, opt, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.down_size = opt.adapter_bottleneck_dim
        self.ln_position = opt.adapter_ln_position
        self.adapter_ln = nn.LayerNorm(self.n_embd)
        self.scale = float(opt.adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = opt.feature_dropout
        
        # IMPROVE ME
        if opt.adapter_init_option == "bert":
            self.apply(init_bert_weights)
        elif opt.adapter_init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        else:
            raise NotImplementedError("Adapter init option not implemented!")
        
    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        
        if self.ln_position == 'pre':
            x = self.adapter_ln(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.ln_position == 'post':
            up = self.adapter_ln(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output

# modified from timm/models/vision_transformer.py
class AdaptBlock(nn.Module):

    def __init__(
            self, 
            opt, 
            dim,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.opt = opt
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adapter = AdapterLayer(self.opt, n_embd=dim)
    
    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if self.opt.adapter_option == 'parallel':
            adapt_x = self.adapter(x, add_residual=False)
        residual = x
        x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if self.opt.adapter_option == 'sequential':
            x = self.adapter(x, add_residual=True)
        elif self.opt.adapter_option == 'parallel':
            x = x + adapt_x
        else:
            raise NotImplementedError
        x = residual + x
        return x
