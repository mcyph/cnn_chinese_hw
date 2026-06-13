"""Sharpness-Aware Minimization (SAM).

Foret, Kleiner, Mobahi & Neyshabur, "Sharpness-Aware Minimization for
Efficiently Improving Generalization", ICLR 2021 (arXiv:2010.01412), with the
scale-invariant ASAM variant of Kwon et al., ICML 2021 (arXiv:2102.11600)
available via ``adaptive=True``.

SAM looks for parameters that sit in *flat* regions of the loss landscape (low
loss in a whole neighbourhood, not just at a point), which correlates with
better generalisation. It does so with a two-step update: a gradient *ascent*
step to the worst-case nearby point, then the real descent step computed there.

This is particularly relevant for this project: ~15k character classes with
only a couple of training examples each is a small-data, easy-to-overfit regime,
exactly where flat-minima methods tend to help. SAM costs two forward/backward
passes per step, so it is opt-in (``TrainConfig.sam``).

Implementation follows the widely-used reference by davda54
(https://github.com/davda54/sam).
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0, "rho should be non-negative"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local worst-case point
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # back to the original point
        self.base_optimizer.step()               # real descent step
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm


def disable_running_stats(model):
    """Freeze BatchNorm running-stat updates (used during SAM's second forward
    so the batch statistics are not counted twice)."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.backup_momentum = m.momentum
            m.momentum = 0.0


def enable_running_stats(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum
