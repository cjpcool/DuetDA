from abc import ABC, abstractmethod
import os
import copy
import logging
from typing import Callable, Tuple, List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import wandb
import torch.nn.functional as F
from collections import defaultdict
from modules.utils import EMAGeneralizationStatus, ema_beta_schedule

logger = logging.getLogger(__name__)




class ModelTrainer(ABC):
    """
    Abstract base class for inner model training.
    Interface must stay unchanged.
    """

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        data_weights: torch.Tensor,
        create_graph: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def compute_validation_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        loss_fn: Callable,
    ) -> torch.Tensor:
        pass




class MetaLearner:
    """
    Train Dual-branch data valuator (data_attributor) via bilevel optimization.

    Interface is kept the same as your original version.
    """

    def __init__(
        self,
        data_attributor: nn.Module,
        model_trainer,
        model_loss_fn: Callable,
        meta_loss_fn: Callable,
        meta_lr: float = 0.001,
        truncation_steps: int = 2,
        reinit_frequency: int = 100,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_path: Optional[str] = "./checkpoints",
        ablation: str = 'full',
    ):
        self.data_attributor = data_attributor
        self.model_trainer = model_trainer
        self.model_loss_fn = model_loss_fn
        self.meta_loss_fn = meta_loss_fn
        self.meta_lr = meta_lr
        self.truncation_steps = truncation_steps
        self.reinit_frequency = reinit_frequency
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Keep these fields for compatibility
        self.per_model_meta_optimizers: List[optim.Optimizer] = []
        self.meta_grad_norms: List[float] = []
        
        # EMA status (initialized in meta_train)
        self._status_trackers: List[EMAGeneralizationStatus] = []
        self._current_status: List[torch.Tensor] = []
        self._status_expected_dim: Optional[int] = None

        self.data_attributor.to(self.device)
        
        self.ablation = ablation

    # --------------------------
    # Small utilities (reduce bugs)
    # --------------------------
    def _move(self, obj):
        """Recursively move tensors inside nested structures to self.device."""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=(self.device.type == "cuda"))
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [self._move(v) for v in obj]
            return type(obj)(out)
        return obj

    def _get_target(self, batch: Dict[str, Any]) -> torch.Tensor:
        t = batch.get("target", batch.get("y"))
        if t is None:
            raise KeyError("Batch must contain key 'target' or 'y'.")
        return t

    def _attributor_forward(self, x: torch.Tensor, branch: str, status_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through the data_attributor.

        Branch semantics:
          - 'iid' / 'physics': return IID scorer only (physics_head)
          - 'ood' / 'gen'    : return OOD scorer only (gen_head)
          - 'combined'       : return gated mixture (uses status_vec)
        """
        if branch in ("physics", "iid"):
            return self.data_attributor(x, weight_phy=1.0, weight_gen=0.0)
        elif branch in ("gen", "ood"):
            return self.data_attributor(x, weight_phy=0.0, weight_gen=1.0)
        elif branch == "combined":
            status = status_vec
            if status is not None:
                status = status.to(self.device)
            return self.data_attributor(x, status=status)
        else:
            raise ValueError(f"Unknown branch: {branch}")

    def _get_weights(self, batch: Dict[str, Any], require_grad: bool, branch: str = "combined", status_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = batch.get("input_attributor", None)
        if x is None:
            raise KeyError("Batch must contain key 'input_attributor' for data_attributor.")
        x = self._move(x)

        if not require_grad:
            with torch.no_grad():
                self.data_attributor.eval()
                r = self._attributor_forward(x, branch, status_vec=status_vec)
        else:
            self.data_attributor.train()
            r = self._attributor_forward(x, branch, status_vec=status_vec)
            
        w = r.view(-1)  # avoid squeeze() shape bugs
        w = w / (w.mean() + 1e-8) # normalize to mean 1, in case w becomes lr finally
        w = w.clamp(0.0, 1.0)
        w = 0.1 + 0.9 * w
        return w

    def _extract_inner_lr(self, optimizer: optim.Optimizer) -> float:
        lr = optimizer.defaults.get("lr", None)
        if lr is None:
            # Fallback to param_group[0]
            lr = optimizer.param_groups[0].get("lr", 1e-3)
        return float(lr)

    class _FunctionalModel(nn.Module):
        """A thin wrapper so ModelTrainer.compute_validation_loss can call model(*input_var) unchanged."""
        def __init__(self, base_model: nn.Module, state: Dict[str, torch.Tensor]):
            super().__init__()
            self.base_model = base_model
            self.state = state

        def forward(self, *args, **kwargs):
            return functional_call(self.base_model, self.state, args, kwargs)

    def _build_state(self, model: nn.Module, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # include buffers so functional_call works with modules using buffers
        buffers = dict(model.named_buffers())
        # params override model params
        state = {}
        state.update(buffers)
        state.update(params)
        return state

    # --------------------------
    # Differentiable inner training
    # --------------------------
    def train_inner_model_differentiable(
        self,
        model: nn.Module,
        train_batches: List[Dict[str, torch.Tensor]],
        model_optimizer: optim.Optimizer,
        num_steps: int = 2,
        branch: str = "combined",
        status_vec: Optional[torch.Tensor] = None
    ) -> Tuple[nn.Module, float]:
        """
        Differentiable inner updates using functional SGD-style steps.

        Interface kept the same: returns (model_like, avg_loss).
        Returned model is a wrapper that behaves like nn.Module for validation loss calls.
        """
        base_model = model.to(self.device)
        base_model.train()

        lr = self._extract_inner_lr(model_optimizer)

        # Start from current parameters (as tensors)
        params = {name: p for name, p in base_model.named_parameters()}
        total_loss = 0.0
        steps = max(int(num_steps), 0)
        if steps == 0:
            # No update: return original model wrapper
            state = self._build_state(base_model, params)
            return self._FunctionalModel(base_model, state), 0.0

        for step, raw_batch in enumerate(train_batches[:steps]):
            batch = self._move(raw_batch)

            weights = self._get_weights(batch, require_grad=True, branch=branch, status_vec=status_vec)  # [B]
            input_var = batch["input"]
            target = self._get_target(batch)

            # Forward with current params
            state = self._build_state(base_model, params)
            fmodel = self._FunctionalModel(base_model, state)
            output = fmodel(*input_var)
            
            # Ensure consistent dimensions for regression
            # CGCNN outputs [B, 1], squeeze to [B] to match target
            if output.dim() > 1 and output.size(-1) == 1:
                output = output.squeeze(-1)  # [B, 1] -> [B]
            if target.dim() > 1:
                target = target.squeeze(-1)  # [B, 1] -> [B]

            loss_raw = self.model_loss_fn(output, target)

            # Make a per-sample loss vector if possible
            if loss_raw.ndim == 0:
                # Scalar loss: cannot do true per-sample weighting
                weighted_loss = loss_raw * weights.mean()
                loss_mean = float(loss_raw.detach().item())
            else:
                # Reduce extra dims to get [B]
                loss_per_sample = loss_raw.view(loss_raw.size(0), -1).mean(dim=1)
                if loss_per_sample.numel() != weights.numel():
                    raise ValueError(
                        f"Weight shape {tuple(weights.shape)} mismatches loss_per_sample {tuple(loss_per_sample.shape)}"
                    )
                weighted_loss = (loss_per_sample * weights).mean()
                loss_mean = float(loss_per_sample.mean().detach().item())

            total_loss += loss_mean

            # Compute grads w.r.t. current params (create_graph=True for meta-grad)
            grads = torch.autograd.grad(
                weighted_loss,
                list(params.values()),
                create_graph=True,
                allow_unused=False,
            )

            # SGD-style functional update
            new_params = {}
            for (name, p), g in zip(params.items(), grads):
                new_params[name] = p - lr * g
            params = new_params

        avg_loss = total_loss / steps

        # Return a functional wrapper model carrying the final params
        final_state = self._build_state(base_model, params)
        return self._FunctionalModel(base_model, final_state), avg_loss

    # --------------------------
    # Standard inner training (non-diff), simplified and correct num_steps behavior
    # --------------------------
    def train_inner_model_standard(
        self,
        model: nn.Module,
        train_batch,
        model_optimizer: optim.Optimizer,
        branch: str = 'combined',
        status_vec: Optional[torch.Tensor] = None
    ) -> nn.Module:
        model = model.to(self.device)
        model.train()

        batch = self._move(train_batch)
        weights = self._get_weights(batch, require_grad=False, branch=branch, status_vec=status_vec)  # [B]

        loss, _, model_grad_norm = self.model_trainer.train_step(
            model=model,
            batch=batch,
            loss_fn=self.model_loss_fn,
            optimizer=model_optimizer,
            data_weights=weights,
            create_graph=False,
        )

        return model, loss, model_grad_norm

    # --------------------------
    # Meta-gradients (physics/gen branches)
    # --------------------------
    def _avg_val_loss(self, model: nn.Module, val_batches: List[Dict[str, Any]]) -> torch.Tensor:
        model.eval()
        total = 0.0
        for raw_batch in val_batches:
            batch = self._move(raw_batch)
            total = total + self.model_trainer.compute_validation_loss(
                model=model,
                batch=batch,
                loss_fn=self.meta_loss_fn,
            )
        return total / max(len(val_batches), 1)

    def compute_meta_gradient_physics_branch(
        self,
        model: nn.Module,
        val_batches: List[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        avg_val_loss = self._avg_val_loss(model, val_batches)

        physics_params = list(self.data_attributor.physics_head.parameters())
        grads = torch.autograd.grad(
            avg_val_loss, physics_params, create_graph=False, allow_unused=True
        )

        grad_dict: Dict[str, torch.Tensor] = {}
        for i, (p, g) in enumerate(zip(physics_params, grads)):
            key = f"physics_head.param_{i}"
            grad_dict[key] = (g.detach().clone() if g is not None else torch.zeros_like(p))

        return grad_dict, float(avg_val_loss.detach().item())

    def compute_meta_gradient_gen_branch(
        self,
        model: nn.Module,
        val_batches: List[Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        avg_val_loss = self._avg_val_loss(model, val_batches)

        gen_params = list(self.data_attributor.gen_head.parameters())
        grads = torch.autograd.grad(
            avg_val_loss, gen_params, create_graph=False, allow_unused=True, retain_graph=False
        )

        grad_dict: Dict[str, torch.Tensor] = {}
        for i, (p, g) in enumerate(zip(gen_params, grads)):
            key = f"gen_head.param_{i}"
            grad_dict[key] = (g.detach().clone() if g is not None else torch.zeros_like(p))

        return grad_dict, float(avg_val_loss.detach().item())


    @torch.no_grad()
    def _concat_attributor_inputs(self, batches, max_batches: int = 1):
        """Utility: gather and concat input_attributor from a few batches (no grad)."""
        xs = []
        for raw_batch in batches[:max(int(max_batches), 1)]:
            b = self._move(raw_batch)
            x = b.get("input_attributor", None)
            if x is None:
                raise KeyError("Batch must contain key 'input_attributor' for decorrelation penalty.")
            xs.append(x)
        return torch.cat(xs, dim=0)


    def compute_meta_gradient_gate(
        self,
        model: nn.Module,
        iid_val_batches: List[Dict[str, torch.Tensor]],
        ood_val_batches: List[Dict[str, torch.Tensor]],
        lambda_ood: float = 1.0,
    ) -> Tuple[Dict[str, torch.Tensor], float, float, float]:
        iid_loss = self._avg_val_loss(model, iid_val_batches)
        ood_loss = self._avg_val_loss(model, ood_val_batches)
        gate_loss = iid_loss + lambda_ood * ood_loss

        gate_params = list(self.data_attributor.gate.parameters())
        grads = torch.autograd.grad(gate_loss, gate_params, create_graph=False, allow_unused=True)

        grad_dict = {}
        for i, (p, g) in enumerate(zip(gate_params, grads)):
            grad_dict[f"gate.param_{i}"] = (g.detach().clone() if g is not None else torch.zeros_like(p))

        return grad_dict, float(gate_loss.detach().item()), float(iid_loss.detach().item()), float(ood_loss.detach().item())


    def compute_decorrelation_grads(
        self,
        batches,                       # usually train_batches
        lambda_corr: float = 1e-3,
        penalty: str = "corr",         # "corr" or "cov"
        max_batches: int = 1,          # use >1 to reduce noise
        eps: float = 1e-6,
        # --- anti-collapse knobs ---
        var_floor: float = 0.0,        # >0 : enforce minimum variance for each head
        mean_floor: float = 0.0,       # >0 : enforce mean not too close to 0 (prevents "all-zero")
        edge_floor: float = 0.0,       # >0 : penalize being too close to 0/1 (prevents saturation)
        lambda_var: float = 1.0,       # weight for var_floor penalty (relative to decorrelation term)
        lambda_mean: float = 1.0,      # weight for mean_floor penalty
        lambda_edge: float = 1.0,      # weight for edge_floor penalty
        detach_inputs: bool = True,    # usually True; inputs don't need grad
    ):
        """
        Returns:
            grad_phy_dict, grad_gen_dict, reg_scaled(float), stats(dict)
        Where grad_*_dict keys match your existing grad dict keys.
        """

        # 1) build x (optionally no-grad)
        if detach_inputs:
            x = self._concat_attributor_inputs(batches, max_batches=max(int(max_batches), 1))
            x = x.detach()
        else:
            xs = []
            for raw_batch in batches[:max(int(max_batches), 1)]:
                b = self._move(raw_batch)
                xs.append(b["input_attributor"])
            x = torch.cat(xs, dim=0)

        # 2) forward each head (IMPORTANT: use head outputs directly, not combined)
        v_phy = self.data_attributor.physics_head(x).view(-1)
        v_gen = self.data_attributor.gen_head(x).view(-1)

        # 3) decorrelation penalty term
        vp = v_phy - v_phy.mean()
        vg = v_gen - v_gen.mean()

        cov = (vp * vg).mean()
        if penalty == "corr":
            var_p = vp.var(unbiased=False) + eps
            var_g = vg.var(unbiased=False) + eps
            rho = cov / torch.sqrt(var_p * var_g)
            decorr = rho * rho
        elif penalty == "cov":
            decorr = cov * cov
        else:
            raise ValueError(f"Unknown penalty='{penalty}', choose from ['corr','cov'].")

        # 4) anti-collapse penalties (very important in your setting)
        # 4.1 variance floor: prevents constant outputs
        var_phy = v_phy.var(unbiased=False)
        var_gen = v_gen.var(unbiased=False)
        var_pen = 0.0
        if var_floor > 0.0:
            var_pen = F.relu(var_floor - var_phy) + F.relu(var_floor - var_gen)

        # 4.2 mean floor: prevents "all close to 0" collapse
        # NOTE: since v in [0,1], collapse-to-0 typically appears as mean -> 0
        mean_phy = v_phy.mean()
        mean_gen = v_gen.mean()
        mean_pen = 0.0
        if mean_floor > 0.0:
            mean_pen = F.relu(mean_floor - mean_phy) + F.relu(mean_floor - mean_gen)

        # 4.3 edge penalty: discourage being too close to 0 or 1 (optional)
        # edge_floor means keep values away from [0, edge_floor] and [1-edge_floor, 1]
        edge_pen = 0.0
        if edge_floor > 0.0:
            # penalize proximity to 0
            edge_pen = edge_pen + F.relu(edge_floor - v_phy).mean() + F.relu(edge_floor - v_gen).mean()
            # penalize proximity to 1
            edge_pen = edge_pen + F.relu(v_phy - (1.0 - edge_floor)).mean() + F.relu(v_gen - (1.0 - edge_floor)).mean()

        # total regularizer (unscaled)
        reg = decorr \
            + float(lambda_var) * var_pen \
            + float(lambda_mean) * mean_pen \
            + float(lambda_edge) * edge_pen

        # scale
        reg_scaled = float(lambda_corr) * reg

        # 5) grads wrt each head params
        physics_params = list(self.data_attributor.physics_head.parameters())
        gen_params = list(self.data_attributor.gen_head.parameters())

        g_phy = torch.autograd.grad(
            reg_scaled, physics_params, create_graph=False, retain_graph=True, allow_unused=True
        )
        g_gen = torch.autograd.grad(
            reg_scaled, gen_params, create_graph=False, retain_graph=False, allow_unused=True
        )

        # 6) pack into dicts (same key format as your meta grads)
        grad_phy_dict = {}
        none_phy = 0
        for i, (p, g) in enumerate(zip(physics_params, g_phy)):
            if g is None:
                none_phy += 1
                g = torch.zeros_like(p)
            grad_phy_dict[f"physics_head.param_{i}"] = g.detach().clone()

        grad_gen_dict = {}
        none_gen = 0
        for i, (p, g) in enumerate(zip(gen_params, g_gen)):
            if g is None:
                none_gen += 1
                g = torch.zeros_like(p)
            grad_gen_dict[f"gen_head.param_{i}"] = g.detach().clone()

        stats = {
            # overall
            "reg": float(reg.detach().item()),
            "decorr": float(decorr.detach().item()),
            "reg_scaled": float(reg_scaled.detach().item()),
            "none_phy": none_phy,
            "none_gen": none_gen,

            # means/vars/stds
            "vphy_mean": float(mean_phy.detach().item()),
            "vgen_mean": float(mean_gen.detach().item()),
            "vphy_var": float(var_phy.detach().item()),
            "vgen_var": float(var_gen.detach().item()),
            "vphy_std": float(v_phy.detach().std(unbiased=False).item()),
            "vgen_std": float(v_gen.detach().std(unbiased=False).item()),

            # collapse diagnostics
            "var_pen": float(var_pen.detach().item()) if isinstance(var_pen, torch.Tensor) else float(var_pen),
            "mean_pen": float(mean_pen.detach().item()) if isinstance(mean_pen, torch.Tensor) else float(mean_pen),
            "edge_pen": float(edge_pen.detach().item()) if isinstance(edge_pen, torch.Tensor) else float(edge_pen),
            "gap_mean": float((mean_gen - mean_phy).detach().item()),
        }

        return grad_phy_dict, grad_gen_dict, float(reg_scaled.detach().item()), stats



    # --------------------------
    # Main meta-training loop (interface unchanged)
    # --------------------------

    # --------------------------
    # Main meta-training loop (interface unchanged)
    # --------------------------
    def meta_train(
        self,
        train_batches,
        iid_val_batches,
        ood_val_batches,
        inner_models: List[nn.Module],
        model_optimizers: List[optim.Optimizer],
        model_factory: Callable,
        num_outer_steps: int = 100,
        inner_steps: int = 5,
    ) -> Dict[str, Any]:

        # One meta optimizer (η for the data_attributor)
        meta_opt = optim.Adam(self.data_attributor.parameters(), lr=self.meta_lr)
        # keep compatibility field
        if not self.per_model_meta_optimizers:
            self.per_model_meta_optimizers = [meta_opt for _ in range(len(inner_models))]

        n_models = max(len(inner_models), 1)
        if len(inner_models) != len(model_optimizers):
            raise ValueError(f"inner_models ({len(inner_models)}) and model_optimizers ({len(model_optimizers)}) must match")

        # Save initial states for periodic reinit
        initial_states = [copy.deepcopy(m.state_dict()) for m in inner_models]

        history = {
            "outer_step": [],
            "meta_grad_norm": [],
            "iid_val_loss": [],
            "ood_val_loss": [],
            "gate_meta_loss": [],
            "train_loss": [],
            "status": [],
        }

        logger.info(f"Starting meta-training for {num_outer_steps} iterations")
        logger.info(f"  - Number of inner models: {len(inner_models)}")
        logger.info(f"  - Truncation steps: {self.truncation_steps}")
        logger.info(f"  - Reinit frequency: {self.reinit_frequency}")
        value_log_dict = defaultdict(float)
        # Initialize wandb (optional)
        if wandb.run is None:
            wandb.init(
                project="data-valuation",
                config={
                    "num_models": len(inner_models),
                    "num_outer_steps": num_outer_steps,
                    "inner_steps": inner_steps,
                    "truncation_steps": self.truncation_steps,
                    "reinit_frequency": self.reinit_frequency,
                    "meta_lr": self.meta_lr,
                },
            )

        # Iterators: keep meta-unroll batches independent from standard inner training batches
        it_meta_train = [iter(train_batches) for _ in range(n_models)]
        it_train = [iter(train_batches) for _ in range(n_models)]
        it_iid_val = [iter(iid_val_batches) for _ in range(n_models)]
        it_ood_val = [iter(ood_val_batches) for _ in range(n_models)]

        # Per-model EMA trackers
        self._status_trackers = []
        for _ in range(n_models):
            self._status_trackers.append(
                EMAGeneralizationStatus(
                    beta_schedule=ema_beta_schedule(steps=self.reinit_frequency, max_beta=0.9),
                    expected_dim=self.data_attributor.gate.status_dim,
                    use_deltas=True,
                    device=next(self.data_attributor.parameters()).device,
                    normalize=True,
                )
            )

        # Epoch-like accumulators used to update the EMA status tracker whenever we wrap it_train
        epoch_idx = [0 for _ in range(n_models)]
        # standard inner training stats
        epoch_train_sum = [0.0 for _ in range(n_models)]
        epoch_train_cnt = [0 for _ in range(n_models)]
        # meta stats (outer)
        epoch_iid_sum = [0.0 for _ in range(n_models)]
        epoch_ood_sum = [0.0 for _ in range(n_models)]
        epoch_mgn_sum = [0.0 for _ in range(n_models)]
        epoch_outer_cnt = [0 for _ in range(n_models)]

        def _next_from(it_list, batches, i: int):
            """Get next batch from iterator; reset on StopIteration."""
            try:
                return next(it_list[i])
            except StopIteration:
                it_list[i] = iter(batches)
                return next(it_list[i])

        def _update_status_tracker(i: int):
            """Update EMA tracker using epoch accumulators (called when train iterator wraps)."""
            if self._status_trackers is None or len(self._status_trackers) != n_models:
                return
            # averages
            tr = epoch_train_sum[i] / max(epoch_train_cnt[i], 1)
            iid = epoch_iid_sum[i] / max(epoch_outer_cnt[i], 1)
            ood = epoch_ood_sum[i] / max(epoch_outer_cnt[i], 1)
            gn = epoch_mgn_sum[i] / max(epoch_outer_cnt[i], 1)
            self._status_trackers[i].update(
                step=epoch_idx[i],
                train_loss=tr,
                iid_val_loss=iid,
                ood_val_loss=ood,
                meta_grad_norm=gn,
            )
            epoch_idx[i] += 1
            # reset accumulators
            epoch_train_sum[i] = 0.0
            epoch_train_cnt[i] = 0
            epoch_iid_sum[i] = 0.0
            epoch_ood_sum[i] = 0.0
            epoch_mgn_sum[i] = 0.0
            epoch_outer_cnt[i] = 0

        def _next_train_batch(i: int):
            """Next training batch; if epoch ends, update status tracker then reset iterator."""
            try:
                return next(it_train[i])
            except StopIteration:
                _update_status_tracker(i)
                it_train[i] = iter(train_batches)
                return next(it_train[i])

        # Convenient param groups
        physics_params = list(self.data_attributor.physics_head.parameters())
        gen_params = list(self.data_attributor.gen_head.parameters())
        gate_params = list(self.data_attributor.gate.parameters())

        def _accum_grads(params, grads):
            for p, g in zip(params, grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.add_(g.detach())

        def _avg_tensor_norm(gs):
            vals = [g.detach().norm(2) for g in gs if g is not None]
            if not vals:
                return 0.0
            return float(torch.stack(vals).mean().item())

        lambda_ood_gate = 1.0

        # --------------------------
        # Outer loop
        # --------------------------
        for k in range(int(num_outer_steps)):
            # periodic reinit (inner-model population stratification)
            if k > 0 and (k % self.reinit_frequency == 0):
                logger.info(f"Reinitializing inner models at step {k}")
                for i, (m, st) in enumerate(zip(inner_models, initial_states)):
                    m.load_state_dict(copy.deepcopy(st))
                    self._status_trackers[i].reset()
                    # also reset iterators & accumulators
                    it_meta_train[i] = iter(train_batches)
                    it_train[i] = iter(train_batches)
                    it_iid_val[i] = iter(iid_val_batches)
                    it_ood_val[i] = iter(ood_val_batches)
                    epoch_idx[i] = 0
                    epoch_train_sum[i] = 0.0
                    epoch_train_cnt[i] = 0
                    epoch_iid_sum[i] = 0.0
                    epoch_ood_sum[i] = 0.0
                    epoch_mgn_sum[i] = 0.0
                    epoch_outer_cnt[i] = 0

            # read current status vectors
            self._current_status = [st.vector().detach() for st in self._status_trackers]
            history["status"].append([s.cpu().tolist() for s in self._current_status])

            # ---- meta-grad accumulation ----
            meta_opt.zero_grad(set_to_none=True)

            per_model_iid = []
            per_model_ood = []
            per_model_gate = []
            per_model_train = []
            per_model_gn = []

            for i, (model, opt) in enumerate(zip(inner_models, model_optimizers)):
                # 1) IID unroll: branch='iid' -> only IID scorer
                train_window_iid = [_next_from(it_meta_train, train_batches, i) for _ in range(self.truncation_steps)]
                iid_val_batch = _next_from(it_iid_val, iid_val_batches, i)

                if self.ablation == 'wo_phy':
                    # skip IID branch
                    iid_val_loss = torch.tensor(0.0, device=self.device)
                    diff_train_iid = 0.0
                    g_phy = []
                else:
                    f_iid, diff_train_iid = self.train_inner_model_differentiable(
                        model=model,
                        train_batches=train_window_iid,
                        model_optimizer=opt,
                        num_steps=self.truncation_steps,
                        branch="iid",
                        status_vec=None,
                    )
                    iid_val_loss = self._avg_val_loss(f_iid, [iid_val_batch])

                    g_phy = torch.autograd.grad(
                        iid_val_loss,
                        physics_params,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    _accum_grads(physics_params, g_phy)

                # 2) OOD unroll: branch='ood' -> only OOD scorer
                train_window_ood = [_next_from(it_meta_train, train_batches, i) for _ in range(self.truncation_steps)]
                ood_val_batch = _next_from(it_ood_val, ood_val_batches, i)
                
                if self.ablation == 'wo_gen':
                    # skip OOD branch
                    ood_val_loss = torch.tensor(0.0, device=self.device)
                    diff_train_ood = 0.0
                    g_gen = []
                else:
                    f_ood, diff_train_ood = self.train_inner_model_differentiable(
                        model=model,
                        train_batches=train_window_ood,
                        model_optimizer=opt,
                        num_steps=self.truncation_steps,
                        branch="ood",
                        status_vec=None,
                    )
                    ood_val_loss = self._avg_val_loss(f_ood, [ood_val_batch])

                    g_gen = torch.autograd.grad(
                        ood_val_loss,
                        gen_params,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    _accum_grads(gen_params, g_gen)

                train_window_gate = [_next_from(it_meta_train, train_batches, i) for _ in range(self.truncation_steps)]
                if self.ablation in ['wo_gate', 'wo_phy', 'wo_gen']:
                    g_gate = []
                    gate_meta_loss = torch.tensor(0.0)
                    diff_train_gate = 0.0
                else:
                    # 3) Gate unroll: branch='combined' (gated mixture) -> update gate params
                    iid_val_batch_g = _next_from(it_iid_val, iid_val_batches, i)
                    ood_val_batch_g = _next_from(it_ood_val, ood_val_batches, i)

                    f_gate, diff_train_gate = self.train_inner_model_differentiable(
                        model=model,
                        train_batches=train_window_gate,
                        model_optimizer=opt,
                        num_steps=self.truncation_steps,
                        branch="combined",
                        status_vec=self._current_status[i],
                    )
                    gate_iid = self._avg_val_loss(f_gate, [iid_val_batch_g])
                    gate_ood = self._avg_val_loss(f_gate, [ood_val_batch_g])
                    gate_meta_loss = gate_iid + lambda_ood_gate * gate_ood

                    g_gate = torch.autograd.grad(
                        gate_meta_loss,
                        gate_params,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    _accum_grads(gate_params, g_gate)

                # 4) Optional head decorrelation regularizer (no unroll needed)
                if self.ablation in ['wo_decorr', 'wo_phy', 'wo_gen']:
                    corr_loss = 0.0
                    corr_stats = {}
                    
                else:
                    corr_g_phy, corr_g_gen, corr_loss, corr_stats = self.compute_decorrelation_grads(
                        batches=train_window_gate,
                        lambda_corr=1e-2,
                        penalty="cov",
                        max_batches=1,
                        var_floor=1e-3,
                        mean_floor=0.05,
                        lambda_var=1.0,
                        lambda_mean=1.0,
                        edge_floor=0.0,
                    )
                    # add corr grads into accumulated grads
                    for idx, p in enumerate(physics_params):
                        g = corr_g_phy.get(f"physics_head.param_{idx}", None)
                        if g is None:
                            continue
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.add_(g)
                    for idx, p in enumerate(gen_params):
                        g = corr_g_gen.get(f"gen_head.param_{idx}", None)
                        if g is None:
                            continue
                        if p.grad is None:
                            p.grad = g.clone()
                        else:
                            p.grad.add_(g)

                # per-model logging
                gn = _avg_tensor_norm(list(g_phy) + list(g_gen) + list(g_gate))
                per_model_iid.append(float(iid_val_loss.detach().item()))
                per_model_ood.append(float(ood_val_loss.detach().item()))
                per_model_gate.append(float(gate_meta_loss.detach().item()))
                per_model_train.append(float((diff_train_iid + diff_train_ood + diff_train_gate) / 3.0))
                per_model_gn.append(gn)

                # update epoch meta accumulators for status tracking
                epoch_iid_sum[i] += float(iid_val_loss.detach().item())
                epoch_ood_sum[i] += float(ood_val_loss.detach().item())
                epoch_mgn_sum[i] += float(gn)
                epoch_outer_cnt[i] += 1

                # (optional) wandb corr logging (per inner model)
                wandb.log({
                    f"corr_loss/model_{i}": corr_loss,
                    f"vphy_std/model_{i}": corr_stats.get("vphy_std", 0.0),
                    f"vgen_std/model_{i}": corr_stats.get("vgen_std", 0.0),
                }, step=k)

            # average grads across models
            for p in self.data_attributor.parameters():
                if p.grad is not None:
                    p.grad.div_(n_models)

            meta_opt.step()

            # --------------------------
            # Standard inner training (advance the population)
            # --------------------------
            step_train_losses = []
            value_cut = 0  # for logging frequency control
            for i, (model, opt) in enumerate(zip(inner_models, model_optimizers)):
                for _ in range(int(inner_steps - self.truncation_steps)):
                    train_batch = _next_train_batch(i)
                    # --------------------------
                    # logging output
                    # --------------------------
                    with torch.no_grad():
                        value_cut += 1
                        batch = train_batch
                        x = batch["input_attributor"]
                        v_phy = self.data_attributor.physics_head(x).view(-1)
                        v_gen = self.data_attributor.gen_head(x).view(-1)
                        value_log_dict["batch/v_phy_mean"] += v_phy.mean().item()
                        value_log_dict["batch/v_gen_mean"] += v_gen.mean().item()
                        

                        # gate output: prefer gate(status); fallback gate(); fallback tensor gate
                        gate_obj = self.data_attributor.gate
                        g_out = None
                        status_vec = self._current_status[i]
                        g_out = gate_obj(status_vec.unsqueeze(0), x)

                        value_log_dict["batch/gate0_mean"] += g_out[0].mean().item()
                        value_log_dict["batch/gate1_mean"] += g_out[1].mean().item()
                        
                    
                    model, loss_std, _gn_std = self.train_inner_model_standard(
                        model=model,
                        train_batch=train_batch,
                        model_optimizer=opt,
                        branch="combined",
                        status_vec=self._current_status[i],
                    )
                    epoch_train_sum[i] += float(loss_std)
                    epoch_train_cnt[i] += 1
                    step_train_losses.append(float(loss_std))

            # --------------------------
            # Logging / checkpoint
            # --------------------------
            value_log_dict = {k: v / max(value_cut, 1) for k, v in value_log_dict.items()}
            wandb.log(value_log_dict, step=k)
            
            avg_iid = sum(per_model_iid) / max(len(per_model_iid), 1)
            avg_ood = sum(per_model_ood) / max(len(per_model_ood), 1)
            avg_gate = sum(per_model_gate) / max(len(per_model_gate), 1)
            avg_train = sum(step_train_losses) / max(len(step_train_losses), 1)
            avg_gn = sum(per_model_gn) / max(len(per_model_gn), 1)

            history["outer_step"].append(k + 1)
            history["meta_grad_norm"].append(avg_gn)
            history["iid_val_loss"].append(avg_iid)
            history["ood_val_loss"].append(avg_ood)
            history["gate_meta_loss"].append(avg_gate)
            history["train_loss"].append(avg_train)

            wandb.log({
                "meta_step": k + 1,
                "avg_meta_grad_norm": avg_gn,
                "avg_train_loss": avg_train,
                "avg_iid_val_loss": avg_iid,
                "avg_ood_val_loss": avg_ood,
                "avg_gate_meta_loss": avg_gate,
            }, step=k)

            if (k + 1) % 50 == 0:
                os.makedirs(self.checkpoint_path, exist_ok=True)
                torch.save(
                    self.data_attributor.state_dict(),
                    os.path.join(self.checkpoint_path, f"data_attributor_meta_step_{k+1}.pt"),
                )
                logger.info(f"Saved data_attributor checkpoint at step {k+1}")

        # finalize tracker for any partially filled epoch
        for i in range(n_models):
            if epoch_train_cnt[i] > 0 or epoch_outer_cnt[i] > 0:
                _update_status_tracker(i)

        logger.info("Meta-training completed")
        history["final_eta"] = self.data_attributor.state_dict()

        # summary logs
        avg_train = sum(history["train_loss"]) / max(len(history["train_loss"]), 1)
        avg_iid = sum(history["iid_val_loss"]) / max(len(history["iid_val_loss"]), 1)
        avg_ood = sum(history["ood_val_loss"]) / max(len(history["ood_val_loss"]), 1)
        logger.info(f"Loss Summary: train={avg_train:.6f}, iid_val={avg_iid:.6f}, ood_val={avg_ood:.6f}")

        wandb.log({
            "final_avg_train_loss": avg_train,
            "final_avg_iid_val_loss": avg_iid,
            "final_avg_ood_val_loss": avg_ood,
        })

        wandb.finish()
        return history
