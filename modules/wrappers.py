import os
import sys
import torch
from torch import nn, optim
from typing import Dict, Callable, Tuple, Optional

from modules.utils import grad_norm_avg

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


from .meta_training import ModelTrainer






class GraphTrainer(ModelTrainer):
    """Generic graph model trainer that works for CGCNN, SchNet, etc."""

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        data_weights: torch.Tensor,
        create_graph: bool = False,
    ) -> Tuple[float, Optional[torch.Tensor]]:

        optimizer.zero_grad(set_to_none=True)

        input_var = batch["input"]
        target = batch.get("target", batch.get("y"))
        if target is None:
            raise KeyError("Batch must contain key 'target' or 'y'.")

        # Forward
        output = model(*input_var)
        
        # Ensure consistent dimensions for regression
        # CGCNN outputs [B, 1], we squeeze to [B] to match typical target shape
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)  # [B, 1] -> [B]
        
        # Ensure target is also [B]
        if target.dim() > 1:
            target = target.squeeze(-1)  # [B, 1] -> [B]

        # Loss
        loss_raw = loss_fn(output, target)
        if data_weights is None:
            w = torch.ones_like(loss_raw).view(-1).to(loss_raw.device, dtype=torch.float32)
        else:   
            w = data_weights.view(-1).to(loss_raw.device, dtype=torch.float32)

        if loss_raw.ndim == 0:
            # scalar loss, fallback to mean weight scaling
            weighted_loss = loss_raw * w.mean()
            loss_scalar = float(loss_raw.detach().item())
        else:
            # reduce to per-sample: [B, ...] -> [B]
            loss_per_sample = loss_raw.view(loss_raw.size(0), -1).mean(dim=1)
            if loss_per_sample.numel() != w.numel():
                raise ValueError(
                    f"data_weights shape {tuple(w.shape)} mismatches loss_per_sample {tuple(loss_per_sample.shape)}"
                )
            weighted_loss = (loss_per_sample * w).mean()
            loss_scalar = float(loss_per_sample.mean().detach().item())

        # Backward
        optimizer.zero_grad()
        weighted_loss.backward(create_graph=create_graph)
        ## obtain gradient norm before step
        grad_norm = grad_norm_avg(model)
        optimizer.step()
        

        return loss_scalar, (weighted_loss if create_graph else None), grad_norm

    def compute_validation_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss_fn: Callable,
    ) -> torch.Tensor:

        input_var = batch["input"]
        target = batch.get("target", batch.get("y"))
        if target is None:
            raise KeyError("Batch must contain key 'target' or 'y'.")

        output = model(*input_var)
        
        # Ensure consistent dimensions for regression
        if output.dim() > 1 and output.size(-1) == 1:
            output = output.squeeze(-1)  # [B, 1] -> [B]
        if target.dim() > 1:
            target = target.squeeze(-1)  # [B, 1] -> [B]
        
        loss = loss_fn(output, target)

        return loss.mean() if loss.ndim > 0 else loss

    

# Backwards compatibility
CGCNNTrainer = GraphTrainer
