"""Small, dependency-free Poincare ball operators used by Hyp-Metric."""
import torch
import torch.nn as nn

def _clamp_c(c):
    return max(float(c), 1e-8)

def project(x, c, eps=1e-5):
    c = _clamp_c(c)
    # Keep the same 1e-3 boundary margin as htdt/hyp_metric.
    r = (1.0 - 1e-3) / c ** 0.5
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x * torch.clamp(r / n, max=1.0)

def expmap0(u, c):
    c = _clamp_c(c)
    sqrt_c = c ** 0.5
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return project(torch.tanh(sqrt_c * n) * u / (sqrt_c * n), c)

def dist_matrix(x, y, c):
    c = _clamp_c(c)
    # Batched (-x) \oplus_c y followed by the source repository's
    # 2/sqrt(c) * artanh(sqrt(c) * ||.||) distance.
    a = -x[:, None, :]
    b = y[None, :, :]
    a2 = (a * a).sum(-1, keepdim=True)
    b2 = (b * b).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    num = (1 + 2*c*ab + c*b2)*a + (1 - c*a2)*b
    den = 1 + 2*c*ab + c*c*a2*b2
    mobius = num / (den + 1e-5)
    arg = (c ** 0.5 * mobius.norm(dim=-1)).clamp(-1+1e-5, 1-1e-5)
    return 2 / c ** 0.5 * torch.atanh(arg)

def dist_matrix_efficient(x, y, c):
    """Equivalent Poincare distance without materializing [N,M,D].

    Intended for retrieval evaluation, where N and M are thousands.  The
    source Mobius implementation above is retained for the training loss so
    its gradients/numerics stay aligned with the original repository.
    """
    c = _clamp_c(c)
    a2 = (x*x).sum(-1,keepdim=True)
    b2 = (y*y).sum(-1,keepdim=True).t()
    ab = -(x @ y.t())
    coef_a = 1 + 2*c*ab + c*b2
    coef_b = 1 - c*a2
    den = 1 + 2*c*ab + c*c*a2*b2
    norm2 = (coef_a.square()*a2 + coef_b.square()*b2
             + 2*coef_a*coef_b*ab).clamp_min(0)
    mobius_norm = norm2.sqrt()/(den + 1e-5)
    arg=(c**.5*mobius_norm).clamp(-1+1e-5,1-1e-5)
    return 2/c**.5*torch.atanh(arg)

class _RiemannianGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = float(c); ctx.save_for_backward(x); return x
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale = (1 - ctx.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale, None

class ToPoincare(nn.Module):
    def __init__(self, c=0.1, clip_r=2.3):
        super().__init__(); self.c = float(c); self.clip_r = float(clip_r)
    def forward(self, x):
        if self.c == 0: return torch.nn.functional.normalize(x, dim=-1)
        if self.clip_r > 0:
            n=x.norm(dim=-1,keepdim=True)+1e-5
            x=x*torch.minimum(torch.ones_like(n), self.clip_r/n)
        return _RiemannianGradient.apply(project(expmap0(x, self.c), self.c), self.c)
