"""
Meta-training script for CGCNN with DualDataValuator.

This script implements the complete meta-training pipeline:
1. Load CGCNN data (train, val, ood, iid splits)
2. Create meta-trainer with DualDataValuator
3. Train data attributor using bilevel optimization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cgcnn.cgcnn.data import CIFData, collate_pool
from cgcnn.cgcnn.model import CrystalGraphConvNet
from modules.data_val import DualDataValuator
from modules.meta_training import MetaLearner
from modules.models.alignn_like import CrystalGraphALIGNN
from modules.models.schnet import CrystalGraphSchNet
from modules.wrappers import CGCNNTrainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/cgcnn_meta_train.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'Random seed set to: {seed}')


def read_ids(path: str) -> List[str]:
    """Read sample IDs from a text file."""
    return [x.strip() for x in open(path).read().splitlines() if x.strip()]


def load_cgcnn_data(
    data_root: str,
    fold: int = 0,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, CIFData]:
    """
    Load CGCNN data from split directories.
    
    Follows the data loading pattern from cgcnn/main.py:
    - Reads CIF files from data_root/cifs
    - Reads splits from data_root/splits/fold[fold]
    - Creates separate dataloaders for train, iid_val, ood_val, test
    
    Args:
        data_root: Root directory containing cifs/ and splits/
        fold: Fold number (default 0)
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, iid_val_loader, ood_val_loader, test_loader, dataset)
    """
    logger.info(f"Loading CGCNN data from {data_root}, fold {fold}")
    
    # Verify split directory structure
    split_dir = os.path.join(data_root, 'splits', f'fold{fold}')
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Load dataset
    dataset = CIFData(split_dir)
    logger.info(f'Loaded dataset with {len(dataset)} samples')
    
    # Create ID to index mapping from id_prop.csv
    id_to_idx = {sid[-1]: i for i, sid in enumerate(dataset)}
    

    def make_subset(ids):
        idx = [id_to_idx['cifs/' + sid] for sid in ids if 'cifs/' + sid in id_to_idx]
        logger.info(f"  Created subset with {len(idx)} samples")
        return Subset(dataset, idx)
    # Read split indices
    train_ids = read_ids(os.path.join(split_dir, "train_candidates.txt"))
    iid_ids = read_ids(os.path.join(split_dir, "iid_val.txt"))
    ood_ids = read_ids(os.path.join(split_dir, "ood_val.txt"))
    test_ids = read_ids(os.path.join(split_dir, "test.txt"))
    
    logger.info(f"Split sizes - Train: {len(train_ids)}, IID Val: {len(iid_ids)}, "
                f"OOD Val: {len(ood_ids)}, Test: {len(test_ids)}")
    
    # Create subsets
    train_set = make_subset(train_ids)
    iid_val_set = make_subset(iid_ids)
    ood_val_set = make_subset(ood_ids)
    test_set = make_subset(test_ids)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pool,
    )
    
    iid_val_loader = DataLoader(
        iid_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pool,
    )
    
    ood_val_loader = DataLoader(
        ood_val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pool,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pool,
    )
    
    logger.info("Data loading complete")
    return train_loader, iid_val_loader, ood_val_loader, test_loader, dataset


def create_inner_models(
    num_models: int,
    orig_atom_fea_len: int,
    nbr_fea_len: int,
    task: str = 'regression',
    atom_fea_len: int = 64,
    h_fea_len: int = 128,
    n_conv: int = 3,
    n_h: int = 1,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    backbone: str = 'cgcnn',
) -> List[nn.Module]:
    """
    Create multiple CGCNN models for meta-training population.
    
    Args:
        num_models: Number of inner models to create
        task: Task type ('regression' or 'classification')
        atom_fea_len: Atom feature length
        h_fea_len: Hidden feature length
        n_conv: Number of convolution layers
        n_h: Number of hidden layers
        device: Device to place models on
        
    Returns:
        List of CGCNN models
    """
    models = []
    backbone = backbone.lower()
    logger.info(f"Creating {num_models} inner {backbone.upper()} models")

    for i in range(num_models):
        if backbone == 'cgcnn':
            model = CrystalGraphConvNet(
                orig_atom_fea_len,
                nbr_fea_len,
                atom_fea_len=args.atom_fea_len,
                n_conv=args.n_conv,
                h_fea_len=args.h_fea_len,
                n_h=args.n_h,
                classification=True if args.task == 'classification' else False,
            )
        elif backbone == 'alignn':
            model = CrystalGraphALIGNN(
                orig_atom_fea_len=orig_atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                node_dim=args.alignn_node_dim,
                edge_dim=args.alignn_edge_dim,
                readout_dim=args.alignn_readout_dim,
                num_layers=args.alignn_layers,
                dropout=args.alignn_dropout,
                classification=True if args.task == 'classification' else False,
            )
        elif backbone == 'schnet':
            model = CrystalGraphSchNet(
                atom_fea_len=orig_atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                hidden_dim=args.schnet_hidden_dim,
                filter_dim=args.schnet_filter_dim,
                num_interactions=args.schnet_num_interactions,
                readout=args.schnet_readout,
                task=args.task,
                num_classes=args.schnet_num_classes,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        model = model.to(device)
        models.append(model)
        logger.debug(f"  Created model {i + 1}/{num_models}")

    return models


def create_model_optimizers(
    models: List[nn.Module],
    optimizer_type: str = 'SGD',
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
) -> List[optim.Optimizer]:
    """
    Create optimizers for inner models.
    
    Args:
        models: List of models to optimize
        optimizer_type: Type of optimizer ('SGD' or 'Adam')
        learning_rate: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: Weight decay
        
    Returns:
        List of optimizers
    """
    logger.info(f"Creating {len(models)} optimizers ({optimizer_type})")
    
    optimizers = []
    for model in models:
        if optimizer_type.upper() == 'SGD':
            opt = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_type.upper() == 'ADAM':
            opt = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        optimizers.append(opt)
    
    return optimizers


def create_data_attributor(
    input_dim: int = 256,
    hidden_dim: int = 128,
    output_dim: int = 1,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
) -> nn.Module:
    """
    Create DualDataValuator as data attributor.
    
    Args:
        input_dim: Input feature dimension (should match model output features)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (importance weight)
        device: Device to place model on
        
    Returns:
        DualDataValuator model
    """
    logger.info(f"Creating DualDataValuator (input_dim={input_dim}, hidden_dim={hidden_dim})")
    data_attributor = DualDataValuator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    data_attributor = data_attributor.to(device)
    
    return data_attributor


def pad_graph_batch(
    atom_fea: torch.Tensor,        # [N_total, Fa]
    nbr_fea: torch.Tensor,         # [N_total, M, Fn]
    nbr_fea_idx: torch.Tensor,     # [N_total, M] (global indices after collate offset)
    crystal_atom_idx: List[torch.Tensor],  # len=B, each [n_i] global indices into N_total
    max_n: int = 20,
) -> torch.Tensor:
    """
    Return: padded & flattened feature for each crystal: [B, D]
    D = max_n * (Fa + M*Fn + M + 1)
        - atom features
        - neighbor features
        - neighbor index features (float, normalized)
        - node mask (1 for real atom, 0 for pad)
    """
    device = atom_fea.device
    B = len(crystal_atom_idx)
    Fa = atom_fea.size(1)
    M = nbr_fea.size(1)
    Fn = nbr_fea.size(2)

    per_crystal_flat = []

    for idx_global in crystal_atom_idx:
        idx_global = idx_global.to(device)
        n_atoms = int(idx_global.numel())
        n_use = min(n_atoms, max_n)

        # pad containers
        atom_pad = atom_fea.new_zeros((max_n, Fa))          # [max_n, Fa]
        nbr_pad  = nbr_fea.new_zeros((max_n, M, Fn))        # [max_n, M, Fn]
        # neighbor idx: keep as float feature later
        idx_pad  = nbr_fea_idx.new_zeros((max_n, M))        # [max_n, M]
        mask     = atom_fea.new_zeros((max_n,))             # [max_n]

        # slice atoms for this crystal (truncate if too many)
        idx_use = idx_global[:n_use]
        atom_pad[:n_use] = atom_fea[idx_use]
        nbr_pad[:n_use]  = nbr_fea[idx_use]
        mask[:n_use] = 1.0

        # nbr idx: convert global -> local
        # in collate_pool, indices are offset by base_idx per crystal, so base = min(idx_global)
        base = int(idx_global.min().item())
        nb_global = nbr_fea_idx[idx_use]                    # [n_use, M]
        nb_local = nb_global - base                         # local indices

        # if we truncated to max_n, neighbors may point outside [0, n_use-1], mask them to 0
        nb_local = nb_local.clamp(min=0)
        nb_local[nb_local >= n_use] = 0

        # store (as local indices)
        idx_pad[:n_use] = nb_local

        # ---- build per-node feature and flatten ----
        # nbr_fea: [max_n, M, Fn] -> [max_n, M*Fn]
        nbr_flat = nbr_pad.reshape(max_n, -1)

        # nbr_idx: [max_n, M] -> float features, normalized (0..1)
        # (better: embedding；先用归一化数值保证能跑通)
        idx_feat = (idx_pad.float() / float(max_n))  # [max_n, M]

        # mask as a feature column
        mask_col = mask.unsqueeze(1)  # [max_n, 1]

        node_feat = torch.cat(
            [atom_pad, nbr_flat, idx_feat, mask_col],
            dim=1
        )  # [max_n, Fa + M*Fn + M + 1]

        per_crystal_flat.append(node_feat.reshape(-1))  # [D]

    return torch.stack(per_crystal_flat, dim=0)  # [B, D]


def prepare_batches_list(loader: DataLoader, device: torch.device, normalizer) -> List[Dict]:
    """
    Convert dataloader to list of batches for meta-training.

    Output batch dict keys:
      - 'input' : CGCNN input tuple (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
      - 'target': normalized target tensor
      - 'input_attributor': [B, D] flattened per-crystal features for MLP attributor
    """
    batches = []

    for i, (inp, target, _) in enumerate(loader):
        # inp should be (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        if not (isinstance(inp, (list, tuple)) and len(inp) == 4):
            raise ValueError(
                f"Expect loader to yield input of length 4 (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx). "
                f"Got type={type(inp)} len={len(inp) if isinstance(inp,(list,tuple)) else 'NA'}"
            )

        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inp

        # move to device
        use_cuda = (args.disable_cuda is False and torch.cuda.is_available())
        if use_cuda:
            atom_fea = atom_fea.to(device, non_blocking=True)
            nbr_fea = nbr_fea.to(device, non_blocking=True)
            nbr_fea_idx = nbr_fea_idx.to(device, non_blocking=True)
            crystal_atom_idx = [c.to(device, non_blocking=True) for c in crystal_atom_idx]
        # keep exact external interface for CGCNN forward
        input_var = (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        # target normalize
        if args.task == "regression":
            target_normed = normalizer.norm(target)
            # Ensure target is [B] shape, not [B, 1]
            if target_normed.dim() > 1:
                target_normed = target_normed.squeeze(-1)
        else:
            target_normed = target.view(-1).long()

        if use_cuda:
            target_var = target_normed.to(device, non_blocking=True)
        else:
            target_var = target_normed

        # --- build per-crystal flattened input for attributor ---
        flatten_input = pad_graph_batch(
            atom_fea=input_var[0],
            nbr_fea=input_var[1],
            nbr_fea_idx=input_var[2],
            crystal_atom_idx=input_var[3],
            max_n=10,          # you can change later
        )
        # Must be [B, D]
        assert flatten_input.size(0) == target_var.size(0), \
            f"flatten_input B={flatten_input.size(0)} but target B={target_var.size(0)}"

        batches.append({
            "input": input_var,
            "target": target_var,
            "input_attributor": flatten_input,
        })

    return batches


def train_meta_learner(
    meta_learner: MetaLearner,
    train_batches: List[Dict],
    iid_val_batches: List[Dict],
    ood_val_batches: List[Dict],
    inner_models: List[nn.Module],
    model_optimizers: List[optim.Optimizer],
    num_outer_steps: int = 50,
    inner_steps: int = 5,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """
    Execute meta-training loop with separate branches for IID and OOD.
    
    Args:
        meta_learner: MetaLearner instance
        train_loader: Training data loader
        iid_val_loader: IID validation data loader (for physics_head)
        ood_val_loader: OOD validation data loader (for gen_head)
        inner_models: List of inner CGCNN models
        model_optimizers: Optimizers for inner models
        num_outer_steps: Number of outer loop iterations
        inner_steps: Number of inner training steps
        device: Device for computation
        
    Returns:
        Training history dictionary
    """
    
    logger.info(f"Starting meta-training: {num_outer_steps} outer steps, {inner_steps} inner steps")
    logger.info(f"  IID validation batches: {len(iid_val_batches)}")
    logger.info(f"  OOD validation batches: {len(ood_val_batches)}")
    
    # Run meta-training
    history = meta_learner.meta_train(
        train_batches=train_batches,
        iid_val_batches=iid_val_batches,
        ood_val_batches=ood_val_batches,
        inner_models=inner_models,
        model_optimizers=model_optimizers,
        model_factory=None,
        num_outer_steps=num_outer_steps,
        inner_steps=inner_steps,
    )
    
    return history


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    data_attributor: nn.Module,
    history: Dict,
):
    """
    Save training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Epoch number
        data_attributor: Data attributor model
        history: Training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'data_attributor_state': data_attributor.state_dict(),
        'history': history,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:04d}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def create_meta_learner(
    data_attributor: nn.Module,
    inner_model_type: str = "cgcnn",
    meta_lr: float = 0.001,
    truncation_steps: int = 2,
    reinit_frequency: int = 100,
    device: torch.device = torch.device("cpu"),
    checkpoint_path: str = './checkpoints/meta_train',
    ablation='full',
) -> MetaLearner:
    """
    Factory function to create a MetaLearner with bilevel optimization.

    Args:
        data_attributor: network φ_η
        inner_model_type: Type of inner model ('cgcnn', 'mlp', etc.)
        meta_lr: Learning rate for per-model meta-optimizers
        truncation_steps: Truncation window for BPTT (default: 2)
        reinit_frequency: Reinitialize models every N steps (default: 100)
        device: Compute device
        checkpoint_path: Path to save checkpoints (default: './checkpoints/meta_train')

    Returns:
        Configured MetaLearner instance
    """
    # Select appropriate inner trainer
    if inner_model_type.lower() in {"cgcnn", "alignn", "schnet"}:
        model_trainer = CGCNNTrainer()
    else:
        raise ValueError(f"Unknown inner model type: {inner_model_type}")

    # Loss functions
    model_loss_fn = nn.MSELoss(reduction="none")
    meta_loss_fn = nn.MSELoss()

    return MetaLearner(
        data_attributor=data_attributor,
        model_trainer=model_trainer,
        model_loss_fn=model_loss_fn,
        meta_loss_fn=meta_loss_fn,
        meta_lr=meta_lr,
        truncation_steps=truncation_steps,
        reinit_frequency=reinit_frequency,
        device=device,
        checkpoint_path=checkpoint_path,
        ablation=ablation,
    )



def main(args):
    """Main training function."""
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load CGCNN data
    train_loader, iid_val_loader, ood_val_loader, test_loader, dataset = load_cgcnn_data(
        data_root=args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    
    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                        'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        from random import sample
        sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)
    
    train_batches = prepare_batches_list(train_loader, device, normalizer)
    iid_val_batches = prepare_batches_list(iid_val_loader, device, normalizer)
    ood_val_batches = prepare_batches_list(ood_val_loader, device, normalizer)
    feat_dim = train_batches[0]['input_attributor'].size(1)
    # Use iid_val as outer loop validation data (meta-gradient computation)
    # and ood_val as held-out test for final evaluation
    
    # Create inner models (CGCNN population)
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    inner_models = create_inner_models(
        num_models=args.num_models,
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        task=args.task,
        atom_fea_len=args.atom_fea_len,
        h_fea_len=args.h_fea_len,
        n_conv=args.n_conv,
        n_h=args.n_h,
        device=device,
        backbone=args.backbone,
    )
    
    # Create optimizers for inner models
    model_optimizers = create_model_optimizers(
        models=inner_models,
        optimizer_type=args.optim,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    # Create data attributor (DualDataValuator)
    # Note: Using h_fea_len as input_dim since that's model output size
    data_attributor = create_data_attributor(
        input_dim=feat_dim,
        hidden_dim=args.attributor_hidden_dim,
        output_dim=1,
        device=device,
    )
    
    # Create meta-learner
    meta_learner = create_meta_learner(
        data_attributor=data_attributor,
        inner_model_type=args.backbone,
        meta_lr=args.meta_lr,
        truncation_steps=args.truncation_steps,
        reinit_frequency=args.reinit_frequency,
        device=device,
        checkpoint_path=args.checkpoint_dir,
        ablation=args.ablation,
    )
    
    logger.info("=" * 80)
    logger.info("META-TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Backbone: {args.backbone}")
    logger.info(f"Num inner models: {args.num_models}")
    if args.backbone == 'cgcnn':
        logger.info(f"Inner model params: atom_fea_len={args.atom_fea_len}, h_fea_len={args.h_fea_len}")
    elif args.backbone == 'alignn':
        logger.info(
            "ALIGNN params: node_dim=%d, edge_dim=%d, readout_dim=%d, layers=%d, dropout=%.3f",
            args.alignn_node_dim,
            args.alignn_edge_dim,
            args.alignn_readout_dim,
            args.alignn_layers,
            args.alignn_dropout,
        )
    elif args.backbone == 'schnet':
        logger.info(
            "SchNet params: hidden_dim=%d, filter_dim=%d, blocks=%d, readout=%s",
            args.schnet_hidden_dim,
            args.schnet_filter_dim,
            args.schnet_num_interactions,
            args.schnet_readout,
        )
    logger.info(f"Meta-learning rate: {args.meta_lr}")
    logger.info(f"Truncation steps: {args.truncation_steps}")
    logger.info(f"Reinit frequency: {args.reinit_frequency}")
    logger.info(f"Outer steps: {args.num_outer_steps}, Inner steps: {args.inner_steps}")
    logger.info("=" * 80)
    
    
    # Train meta-learner
    history = train_meta_learner(
        meta_learner=meta_learner,
        train_batches=train_batches,
        iid_val_batches=iid_val_batches,
        ood_val_batches=ood_val_batches,
        inner_models=inner_models,
        model_optimizers=model_optimizers,
        num_outer_steps=args.num_outer_steps,
        inner_steps=args.inner_steps,
        device=device,
    )
    
    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        epoch=args.num_outer_steps,
        data_attributor=data_attributor,
        history=history,
    )
    
    logger.info("Meta-training complete!")
    
    # Optional: Evaluate on OOD validation set
    if args.eval_ood:
        logger.info("\nEvaluating data attributor on OOD validation set...")
        data_attributor.eval()
        with torch.no_grad():
            ood_attributions = []
            for batch in ood_val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                attr = data_attributor(batch)
                ood_attributions.append(attr.cpu())
        
        logger.info(f"OOD attributions - Mean: {torch.cat(ood_attributions).mean():.4f}, "
                   f"Std: {torch.cat(ood_attributions).std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Meta-training data valuation for CGCNN'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        default='cgcnn_data/matbench_log_kvrh_CountBasedOOD1',
        help='Root directory containing CGCNN data'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=1,
        help='Fold number for cross-validation'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='cgcnn',
        choices=['cgcnn', 'alignn', 'schnet'],
        help='Inner backbone to meta-train (default: cgcnn)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Mini-batch size (default: 64)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=0,
        help='Number of data loading workers (default: 0)'
    )
    
    # Task arguments
    parser.add_argument(
        '--task',
        choices=['regression', 'classification'],
        default='regression',
        help='Task type (default: regression)'
    )
    
    # CGCNN model arguments
    parser.add_argument(
        '--atom-fea-len',
        type=int,
        default=64,
        help='Number of hidden atom features in conv layers'
    )
    parser.add_argument(
        '--h-fea-len',
        type=int,
        default=128,
        help='Number of hidden features after pooling'
    )
    parser.add_argument(
        '--n-conv',
        type=int,
        default=3,
        help='Number of convolutional layers'
    )
    parser.add_argument(
        '--n-h',
        type=int,
        default=1,
        help='Number of hidden layers after pooling'
    )
    parser.add_argument(
        '--alignn-node-dim',
        type=int,
        default=128,
        help='Hidden dimension for ALIGNN node states (default: 128)'
    )
    parser.add_argument(
        '--alignn-edge-dim',
        type=int,
        default=64,
        help='Hidden dimension for ALIGNN edge states (default: 64)'
    )
    parser.add_argument(
        '--alignn-readout-dim',
        type=int,
        default=128,
        help='Readout dimension for ALIGNN crystal head (default: 128)'
    )
    parser.add_argument(
        '--alignn-layers',
        type=int,
        default=4,
        help='Number of ALIGNN message passing layers (default: 4)'
    )
    parser.add_argument(
        '--alignn-dropout',
        type=float,
        default=0.1,
        help='Dropout applied inside ALIGNN layers (default: 0.1)'
    )
    parser.add_argument(
        '--schnet-hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension for SchNet embeddings (default: 128)'
    )
    parser.add_argument(
        '--schnet-filter-dim',
        type=int,
        default=64,
        help='Filter network dimension for SchNet interactions (default: 64)'
    )
    parser.add_argument(
        '--schnet-num-interactions',
        type=int,
        default=3,
        help='Number of SchNet interaction blocks (default: 3)'
    )
    parser.add_argument(
        '--schnet-readout',
        type=str,
        default='mean',
        choices=['mean', 'sum', 'max'],
        help='Pooling strategy for SchNet crystal features (default: mean)'
    )
    parser.add_argument(
        '--schnet-num-classes',
        type=int,
        default=2,
        help='Number of classes for SchNet classification tasks (default: 2)'
    )
    
    # Optimizer arguments
    parser.add_argument(
        '--optim',
        type=str,
        default='Adam',
        choices=['SGD', 'Adam'],
        help='Optimizer for inner models (default: Adam)'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for inner model optimizer (default: 0.01)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for SGD (default: 0.9)'
    )
    parser.add_argument(
        '--weight-decay', '--wd',
        type=float,
        default=0.0,
        help='Weight decay (default: 0.0)'
    )
    
    # Meta-learning arguments
    parser.add_argument(
        '--num-models',
        type=int,
        default=3,
        help='Number of inner models in population (default: 3)'
    )
    parser.add_argument(
        '--meta-lr',
        type=float,
        default=0.001,
        help='Learning rate for data attributor (default: 0.001)'
    )
    parser.add_argument(
        '--truncation-steps',
        type=int,
        default=2,
        help='Truncation steps for BPTT (default: 2)'
    )
    parser.add_argument(
        '--reinit-frequency',
        type=int,
        default=100,
        help='Reinitialize models every N steps (default: 100)'
    )
    parser.add_argument(
        '--attributor-hidden-dim',
        type=int,
        default=16,
        help='Hidden dimension for data attributor (default: 32)'
    )
    
    # Training arguments
    parser.add_argument(
        '--num-outer-steps',
        type=int,
        default=50,
        help='Number of outer loop iterations (default: 50)'
    )
    parser.add_argument(
        '--inner-steps',
        type=int,
        default=5,
        help='Number of inner training steps (default: 5)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--disable-cuda',
        action='store_true',
        help='Disable CUDA'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/meta_train',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--eval-ood',
        action='store_true',
        help='Evaluate on OOD validation set'
    )
    parser.add_argument(
        '--ablation',
        default='full',
        help='Evaluate on OOD validation set'
    )
    

    
    args = parser.parse_args()
    main(args)
'''
python modules/meta_train_cgcnn.py \
  --data_root cgcnn_data/matbench_log_kvrh_CountBasedOOD \
  --checkpoint-dir ./checkpoints/meta_train_q=x/cgcnn \
  --backbone cgcnn \
  --fold 1 \
  --num-models 5 \
  --num-outer-steps 400 \
  --truncation-steps 3 \
  --inner-steps 5 \
  --meta-lr 0.001 \
  --batch-size 256
'''