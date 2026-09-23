"""Backbone registry.

Only the two entry points used by this work are exposed: the plain ViT baseline
and the bias-refinement ViT that carries the query/value adaptation.
"""

import architectures.qv_vit
import architectures.vit


def select(arch, opt):
    if arch == 'vit':
        return architectures.vit.Network(opt)
    if arch == 'qv_vit':
        return architectures.qv_vit.Network(opt)
    raise NotImplementedError('Architecture not implemented.')
