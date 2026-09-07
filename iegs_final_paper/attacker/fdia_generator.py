"""FDIA generator with zero-sum, stealth projection, and post-bound residual checks."""
from __future__ import annotations

import numpy as np
import torch


class FDIAGenerator:
    def __init__(self, power_grid):
        self.power = power_grid
        self.device = self.power.B_matrix.device
        self.B = self.power.B_matrix
        self.B_pinv = self.power.B_pinv

    def _project_dc_stealth(self, z: torch.Tensor) -> torch.Tensor:
        # For DC state estimation, an unobservable vector lies in Range(B).
        return self.B @ (self.B_pinv @ z)

    def check_and_project_fdia(self, z_a, r_d_max: float = 0.20, max_rounds: int = 3):
        if not isinstance(z_a, torch.Tensor):
            z = torch.tensor(z_a, dtype=torch.float32, device=self.device)
        else:
            z = z_a.detach().clone().to(self.device, dtype=torch.float32)

        # Avoid degenerate zero bounds on purely generator buses while still using
        # load-dependent limits for normal buses.
        base_load = self.power.base_load
        mean_load = torch.mean(base_load[base_load > 0]) if torch.any(base_load > 0) else torch.tensor(1.0, device=self.device)
        limit_base = torch.maximum(base_load, 0.05 * mean_load)
        max_tamper = torch.clamp(limit_base * float(r_d_max), min=1e-6)

        # Alternating projection: zero-sum -> stealth subspace -> box.  Recompute
        # residual after the last box projection; the old code measured residual
        # before clipping, which could overstate stealthiness.
        z = z - torch.mean(z)
        for _ in range(max_rounds):
            z = self._project_dc_stealth(z)
            z = z - torch.mean(z)
            z = torch.clamp(z, min=-max_tamper, max=max_tamper)
        z = self._project_dc_stealth(z)
        z = z - torch.mean(z)
        z = torch.clamp(z, min=-max_tamper, max=max_tamper)

        stealth_res = torch.norm(z - self._project_dc_stealth(z)) / (torch.norm(z) + 1e-8)
        zero_sum_res = torch.abs(torch.sum(z)) / (torch.norm(z) + 1e-8)
        max_ratio = torch.max(torch.abs(z) / (limit_base + 1e-8))
        meta = {
            "stealth_residual": float(stealth_res.detach().cpu().item()),
            "zero_sum_residual": float(zero_sum_res.detach().cpu().item()),
            "max_tamper_ratio": float(max_ratio.detach().cpu().item()),
        }
        return z.detach().cpu().numpy(), meta
