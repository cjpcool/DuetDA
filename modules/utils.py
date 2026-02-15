import math
from typing import Optional, Dict, Any, List
import torch



def ema_beta_schedule(steps: int, max_beta: float = 0.9, min_beta: float = 0.0):
    """
    Returns a callable beta(t) in [min_beta, max_beta] that increases with t.
    - small beta early -> responsive EMA
    - larger beta later -> stable EMA
    """
    steps = max(int(steps), 1)

    def beta_fn(t: int) -> float:
        t = max(int(t), 0)
        # cosine ramp from 0 -> 1
        ramp = 0.5 * (1.0 - math.cos(math.pi * min(t, steps) / steps))
        return float(min_beta + (max_beta - min_beta) * ramp)

    return beta_fn


class EMAGeneralizationStatus:
    """
    Tracks EMA of (train loss, iid val loss, ood val loss, meta grad norm),
    plus EMA of derived gap and optional deltas.

    Produces a fixed vector:
      [ema_train_loss,
       ema_iid_val_loss,
       ema_gap,
       ema_meta_grad_norm,
       ema_train_loss_delta (optional),
       ema_gap_delta (optional),
       ... padding ...]
    """
    def __init__(
        self,
        beta_schedule,
        expected_dim: int,
        use_deltas: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
        normalize: bool = True,
    ):
        self.beta_schedule = beta_schedule
        self.expected_dim = int(expected_dim)
        self.use_deltas = bool(use_deltas)
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.normalize = normalize

        # EMA values
        self.ema_train_loss = None
        self.ema_iid = None
        self.ema_ood = None
        self.ema_gap = None
        self.ema_meta_gn = None

        # EMA magnitudes for crude normalization
        self._abs_train = 1.0
        self._abs_iid = 1.0
        self._abs_gap = 1.0
        self._abs_gn = 1.0

        # For deltas (trend)
        self.prev_ema_train = None
        self.prev_ema_gap = None
        self.ema_train_delta = None
        self.ema_gap_delta = None
        self._abs_train_delta = 1.0
        self._abs_gap_delta = 1.0

        self.step = 0
        self._initialized = False

    def reset(self):
        self.__init__(
            beta_schedule=self.beta_schedule,
            expected_dim=self.expected_dim,
            use_deltas=self.use_deltas,
            device=self.device,
            dtype=self.dtype,
            eps=self.eps,
            normalize=self.normalize,
        )

    @staticmethod
    def _to_float(x: Any) -> float:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach()
            if x.numel() != 1:
                x = x.mean()
            return float(x.item())
        return float(x)

    @torch.no_grad()
    def update(
        self,
        step: int,
        train_loss: Any,
        iid_val_loss: Any,
        ood_val_loss: Any,
        meta_grad_norm: Any,
    ):
        self.step = int(step)
        beta = float(self.beta_schedule(self.step))

        train_loss = self._to_float(train_loss)
        iid = self._to_float(iid_val_loss)
        ood = self._to_float(ood_val_loss)
        gn = self._to_float(meta_grad_norm)

        gap = (ood - iid) if (ood is not None and iid is not None) else None

        if not self._initialized:
            # init
            self.ema_train_loss = train_loss
            self.ema_iid = iid
            self.ema_ood = ood
            self.ema_gap = gap
            self.ema_meta_gn = gn

            self._abs_train = abs(train_loss) + self.eps
            self._abs_iid = abs(iid) + self.eps
            self._abs_gap = abs(gap) + self.eps
            self._abs_gn = abs(gn) + self.eps

            # init deltas as 0
            self.prev_ema_train = self.ema_train_loss
            self.prev_ema_gap = self.ema_gap
            self.ema_train_delta = 0.0
            self.ema_gap_delta = 0.0
            self._abs_train_delta = self.eps
            self._abs_gap_delta = self.eps

            self._initialized = True
            return

        # store previous for delta computation (use EMA-to-EMA change)
        prev_train = self.ema_train_loss
        prev_gap = self.ema_gap

        # EMA updates
        self.ema_train_loss = beta * self.ema_train_loss + (1.0 - beta) * train_loss
        self.ema_iid = beta * self.ema_iid + (1.0 - beta) * iid
        self.ema_ood = beta * self.ema_ood + (1.0 - beta) * ood
        self.ema_gap = beta * self.ema_gap + (1.0 - beta) * gap
        self.ema_meta_gn = beta * self.ema_meta_gn + (1.0 - beta) * gn

        # abs trackers (for normalization)
        self._abs_train = beta * self._abs_train + (1.0 - beta) * abs(self.ema_train_loss)
        self._abs_iid = beta * self._abs_iid + (1.0 - beta) * abs(self.ema_iid)
        self._abs_gap = beta * self._abs_gap + (1.0 - beta) * abs(self.ema_gap)
        self._abs_gn = beta * self._abs_gn + (1.0 - beta) * abs(self.ema_meta_gn)

        # optional deltas: EMA of (current_ema - prev_ema)
        if self.use_deltas:
            d_train = self.ema_train_loss - prev_train
            d_gap = self.ema_gap - prev_gap

            self.ema_train_delta = beta * self.ema_train_delta + (1.0 - beta) * d_train
            self.ema_gap_delta = beta * self.ema_gap_delta + (1.0 - beta) * d_gap

            self._abs_train_delta = beta * self._abs_train_delta + (1.0 - beta) * abs(self.ema_train_delta)
            self._abs_gap_delta = beta * self._abs_gap_delta + (1.0 - beta) * abs(self.ema_gap_delta)

    @torch.no_grad()
    def vector(self) -> torch.Tensor:
        if not self._initialized:
            v = torch.zeros(self.expected_dim, device=self.device, dtype=self.dtype)
            return v

        # raw components
        comps = [
            self.ema_train_loss,
            self.ema_iid,
            self.ema_gap,
            self.ema_meta_gn,
        ]

        if self.use_deltas:
            comps += [self.ema_train_delta, self.ema_gap_delta]

        # normalize each dimension roughly to O(1) (optional but recommended)
        if self.normalize:
            scales = [
                self._abs_train,
                self._abs_iid,
                self._abs_gap,
                self._abs_gn,
            ]
            if self.use_deltas:
                scales += [self._abs_train_delta + self.eps, self._abs_gap_delta + self.eps]

            comps = [c / (s + self.eps) for c, s in zip(comps, scales)]

        v = torch.tensor(comps, device=self.device, dtype=self.dtype)

        # pad/truncate to expected_dim
        if v.numel() < self.expected_dim:
            pad = torch.zeros(self.expected_dim - v.numel(), device=v.device, dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        else:
            v = v[: self.expected_dim]
        return v






@torch.no_grad()
def grad_norm_avg(model):
    norms = [p.grad.detach().norm(2).item() for p in model.parameters() if p.grad is not None]
    return sum(norms) / len(norms) if norms else 0.0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = float(val.item())
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        