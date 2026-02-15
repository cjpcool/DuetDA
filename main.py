import torch
import torch.nn as nn
import time
import numpy as np
import argparse

import wandb

from cgcnn.cgcnn.model import CrystalGraphConvNet
from config import CGCNNConfig
from modules.models import CrystalGraphSchNet, CrystalGraphALIGNN

import os
import shutil
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd


from cgcnn.cgcnn.data import CIFData, collate_pool, collate_pool_val, collate_pool_val_with_meta
from modules.data_select import DuetDALoader
from modules.data_val import DualDataValuator
import warnings
from random import sample
from sklearn import metrics
from modules.utils import EMAGeneralizationStatus, ema_beta_schedule
import sys

from modules.utils import AverageMeter, grad_norm_avg
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

warnings.filterwarnings("ignore")


def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

parser = argparse.ArgumentParser(
    description="Compute embeddings with automatic task and data handling"
)
parser.add_argument('--model-name', type=str, default='cgcnn',
                    help='Model name (default: cgcnn)')
parser.add_argument('--data-name', type=str, default='cgcnn_matbench',
                    help='Dataset name (default: cgcnn_matbench)')
parser.add_argument('--data-root', type=str, default='cgcnn_data/matbench_log_kvrh_difficultyOOD1',
                    help='Root directory for dataset (default: cgcnn_data/matbench_log_kvrh_difficultyOOD1)')
parser.add_argument('--fold', type=int, default=1,
                    help='Fold number for dataset splits (default: 1)')
parser.add_argument('--da-method', type=str, default='duetda', choices=['duetda'],
                    help='Data attribution method (only duetda is supported)')
# parser.add_argument('--task', type=str, default='regression',
#                     help='Task type: regression or classification (default: regression)')
parser.add_argument('--cuda', action='store_true',
                    help='Use CUDA if available (default: False)')
parser.add_argument('--task-type', type=str, default='regression',
                    help='Task type: regression or classification (default: regression)')

# SchNet-specific hyperparameters
parser.add_argument('--schnet-hidden-dim', type=int, default=128,
                    help='Hidden dimension for SchNet backbone (default: 128)')
parser.add_argument('--schnet-filter-dim', type=int, default=64,
                    help='Filter network dimension inside SchNet (default: 64)')
parser.add_argument('--schnet-num-interactions', type=int, default=3,
                    help='Number of interaction blocks in SchNet (default: 3)')
parser.add_argument('--schnet-readout', type=str, choices=['mean', 'sum', 'max'], default='mean',
                    help='Pooling strategy for SchNet crystal embeddings (default: mean)')
parser.add_argument('--schnet-num-classes', type=int, default=2,
                    help='Number of classes for SchNet classification tasks (default: 2)')

# ALIGNN-specific hyperparameters
parser.add_argument('--alignn-node-dim', type=int, default=128,
                    help='Hidden dimension for ALIGNN node states (default: 128)')
parser.add_argument('--alignn-edge-dim', type=int, default=64,
                    help='Hidden dimension for ALIGNN edge states (default: 64)')
parser.add_argument('--alignn-readout-dim', type=int, default=128,
                    help='Readout dimension for ALIGNN crystal head (default: 128)')
parser.add_argument('--alignn-layers', type=int, default=4,
                    help='Number of ALIGNN message passing layers (default: 4)')
parser.add_argument('--alignn-dropout', type=float, default=0.1,
                    help='Dropout applied inside ALIGNN blocks (default: 0.1)')

parser.add_argument('--da-input-dim', type=int, default=5970,
                    help='Input dimension for Data Valuator (default: 5970)')
parser.add_argument('--da-model-ckpt', type=str, default="checkpoints/data_attributor_meta_step_500.pt",
                    help='Path to pretrained Data Valuator model checkpoint')

parser.add_argument('--selection-ratio', type=float, default=0.5,
                    help='Selection ratio for DuetDALoader (default: 0.7)')

parser.add_argument('--batch-size', type=int, default=256,
                    help='Mini-batch size (default: 256)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for SGD optimizer (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay (L2 penalty) (default: 0.0)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='Optimizer type: SGD or Adam (default: SGD)')  
parser.add_argument('--workers', type=int, default=0,
                    help='Number of data loading workers (default: 0)')
parser.add_argument('--epochs', type=int, default=0,
                    help='Number of total epochs to run (default: 100)')
parser.add_argument('--lr-milestones', type=int, nargs='+', default=[50, 75],
                    help='Milestones for MultiStepLR (default: [50, 75])')

parser.add_argument('--print-freq', type=int, default=10,
                    help='Print frequency (default: 10)')
parser.add_argument('--resume', type=str, default='',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                    help='Directory for saving/loading checkpoints (default: checkpoints)')
parser.add_argument('--test-res-path', type=str, default=None,
                    help='Path to save/load test prediction CSV (default: auto-named)')
parser.add_argument('--use-wandb', action='store_true',
                    help='Use Weights & Biases for logging (default: False)')   
parser.add_argument('--wandb-project', type=str, default='cgcnn',
                    help='W&B project name (default: cgcnn)')   
parser.add_argument('--wandb-entity', type=str, default=None,
                    help='W&B entity name (default: None)')
parser.add_argument('--wandb-mode', type=str, default='online',
                    help='W&B mode: online, offline, disabled (default: online)')
parser.add_argument('--wandb-name', type=str, default=None,
                    help='W&B run name (default: None)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')

args = parser.parse_args()
args.checkpoint_dir = os.path.abspath(os.path.expanduser(args.checkpoint_dir))
default_test_results = f"{args.da_method}_{args.selection_ratio}_{args.data_name}_test_results.csv"
test_res_path = args.test_res_path or default_test_results
args.test_res_path = os.path.abspath(os.path.expanduser(test_res_path))





class MultiFlopsMeter:
    """
    Estimate FLOPs by profiling forward once and multiplying by step counts.
    - base model: usually forward+backward => multiplier ~ 3
    - valuator: often forward-only => multiplier ~ 1 (or 3 if also trained)
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.base_forward_flops = None
        self.val_forward_flops = None

        self.base_steps = 0
        self.val_steps = 0

    @torch.no_grad()
    def profile_base_once(self, model, base_input_args):
        """
        base_input_args: tuple/list of args used in `output = model(*input_var)`
        """
        if self.base_forward_flops is not None:
            return
        if FlopCountAnalysis is None:
            return
        model.eval()
        flops = FlopCountAnalysis(model, base_input_args)
        self.base_forward_flops = int(flops.total())
        model.train()

    @torch.no_grad()
    def profile_val_once(self, module_or_fn, val_inputs):
        if self.val_forward_flops is not None:
            return
        if FlopCountAnalysis is None:
            return

        if isinstance(module_or_fn, torch.nn.Module):
            wrap = module_or_fn
        else:
            class _Wrap(torch.nn.Module):
                def __init__(self, fn):
                    super().__init__()
                    self.fn = fn
                def forward(self, *args):
                    return self.fn(*args)
            wrap = _Wrap(module_or_fn)

        wrap = wrap.eval().to(self.device)

        # 关键：detach 所有 Tensor inputs，避免 requires_grad 常量问题
        def _detach(x):
            return x.detach() if torch.is_tensor(x) else x
        val_inputs = tuple(_detach(x) for x in val_inputs)

        flops = FlopCountAnalysis(wrap, val_inputs)
        self.val_forward_flops = int(flops.total())

    def step_base(self, n=1):
        self.base_steps += n

    def step_val(self, n=1):
        self.val_steps += n

    def total_flops(self, base_multiplier=3.0, val_multiplier=1.0):
        base = 0 if self.base_forward_flops is None else base_multiplier * self.base_forward_flops * self.base_steps
        val  = 0 if self.val_forward_flops  is None else val_multiplier  * self.val_forward_flops  * self.val_steps
        return {
            "base_forward_flops": self.base_forward_flops,
            "val_forward_flops": self.val_forward_flops,
            "base_steps": self.base_steps,
            "val_steps": self.val_steps,
            "base_total_flops": int(base),
            "val_total_flops": int(val),
            "total_flops": int(base + val),
        }

class ValuatorWrap(torch.nn.Module):
    def __init__(self, valuator, gate=None):
        super().__init__()
        self.valuator = valuator
        self.gate = gate

    def forward(self, feat, weight_phy, weight_gen, status):
        return self.valuator(feat, weight_phy, weight_gen, status)





def weighted_mse_loss(pred, target, w=None, eps=1e-12, normalize=False, ret_per_sample=False):
    """
    pred/target: [B, ...]
    w:           [B] or broadcastable to per-sample loss
    """
    # per-element squared error
    se = (pred - target) ** 2  # [B, ...]
    # per-sample MSE (average over non-batch dims)
    per_sample = se.flatten(1).mean(dim=1)  # [B]
    if w is None:
        w = torch.ones_like(per_sample)
    else:
        w = w.view(-1).to(per_sample.device)  # [B]

    ret =  (w * per_sample)
    
    return ret if ret_per_sample else ret.mean()


def initialize_model(model_name, task_type='regression'):
    name = (model_name or "cgcnn").lower()
    if name == "cgcnn":
        model = CrystalGraphConvNet(
            CGCNNConfig.orig_atom_fea_len,
            CGCNNConfig.nbr_fea_len,
            atom_fea_len=CGCNNConfig.atom_fea_len,
            n_conv=CGCNNConfig.n_conv,
            h_fea_len=CGCNNConfig.h_fea_len,
            n_h=CGCNNConfig.n_h,
            classification=(task_type == 'classification'),
        )
    elif name == "schnet":
        model = CrystalGraphSchNet(
            atom_fea_len=CGCNNConfig.orig_atom_fea_len,
            nbr_fea_len=CGCNNConfig.nbr_fea_len,
            hidden_dim=args.schnet_hidden_dim,
            filter_dim=args.schnet_filter_dim,
            num_interactions=args.schnet_num_interactions,
            readout=args.schnet_readout,
            task=task_type,
            num_classes=args.schnet_num_classes,
        )
    elif name == "alignn":
        model = CrystalGraphALIGNN(
            orig_atom_fea_len=CGCNNConfig.orig_atom_fea_len,
            nbr_fea_len=CGCNNConfig.nbr_fea_len,
            node_dim=args.alignn_node_dim,
            edge_dim=args.alignn_edge_dim,
            readout_dim=args.alignn_readout_dim,
            num_layers=args.alignn_layers,
            dropout=args.alignn_dropout,
            classification=(task_type == 'classification'),
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def initialize_optimizer(optim_name, model, lr, momentum, weight_decay):
    if optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
    elif optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")
    return optimizer





def read_ids(p):
    return [x.strip() for x in open(p).read().splitlines() if x.strip()]


def prepare_matbench_splits(data_root, fold):
    """Load MatBench CGCNN data (symlinks, id_prop.csv, subsets)."""
    cif_dir = os.path.join(data_root, 'cifs')
    split_dir = os.path.join(data_root, 'splits', f"fold{fold}")
    print('CIF data dir:', cif_dir)
    print('Split dir:', split_dir)

    atom_init_path = os.path.join(split_dir, 'atom_init.json')
    if not os.path.exists(atom_init_path):
        root_atom_init = os.path.join(data_root, 'atom_init.json')
        if os.path.exists(root_atom_init):
            shutil.copy2(root_atom_init, atom_init_path)
            print('Copied atom_init.json from root to split_dir')
        else:
            print('Warning: atom_init.json not found in root directory')

    cifs_symlink = os.path.join(split_dir, 'cifs')
    desired_target = os.path.abspath(cif_dir)
    if os.path.islink(cifs_symlink):
        current_target = os.path.realpath(cifs_symlink)
        if current_target != desired_target:
            os.unlink(cifs_symlink)
            os.symlink(desired_target, cifs_symlink, target_is_directory=True)
            print(f'Recreated directory symlink: cifs -> {desired_target}')
        else:
            print('cifs link already exists; skipping')
    elif not os.path.exists(cifs_symlink):
        os.symlink(desired_target, cifs_symlink, target_is_directory=True)
        print(f'Created directory symlink: cifs -> {desired_target}')
    else:
        print('cifs path exists and is not a symlink; leaving as is')

    id_prop_path = os.path.join(split_dir, 'id_prop.csv')
    id_prop_original = id_prop_path + '.original'
    if not os.path.exists(id_prop_original):
        shutil.copy2(id_prop_path, id_prop_original)
        with open(id_prop_original, 'r') as f_in, open(id_prop_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    if not parts[0].startswith('cifs/'):
                        parts[0] = 'cifs/' + parts[0]
                    f_out.write(','.join(parts) + '\n')
        print('Modified id_prop.csv with cifs/ prefix')

    dataset = CIFData(split_dir)
    print(f'Loaded dataset with {len(dataset)} samples')

    id_to_idx = {sid[-1]: i for i, sid in enumerate(dataset)}

    def make_subset(ids):
        idx = [id_to_idx['cifs/' + sid] for sid in ids if 'cifs/' + sid in id_to_idx]
        return Subset(dataset, idx)

    train_ids = read_ids(os.path.join(split_dir, "train_candidates.txt"))
    iid_ids = read_ids(os.path.join(split_dir, "iid_val.txt"))
    ood_ids = read_ids(os.path.join(split_dir, "ood_val.txt"))
    test_ids = read_ids(os.path.join(split_dir, "test.txt"))

    print(train_ids[:5])
    print(list(id_to_idx.items())[:5])

    train_set = make_subset(train_ids)
    iid_val_set = make_subset(iid_ids)
    ood_val_set = make_subset(ood_ids)
    test_set = make_subset(test_ids)  
    

    return dataset, train_set, iid_val_set, ood_val_set, test_set




def initialize_dataset(data_name, data_root, fold):
    if data_name == "cgcnn_matbench":
        dataset, train_set, iid_val_set, ood_val_set, test_set = prepare_matbench_splits(data_root, fold)
               
    return train_set, iid_val_set, ood_val_set, test_set




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
        
def initialize_normalizer(dataset, task_type):
    if task_type == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    return normalizer



def _to_device_cgcnn_input(input, device):
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input[:4]
    return (
        atom_fea.to(device, non_blocking=True),
        nbr_fea.to(device, non_blocking=True),
        nbr_fea_idx.to(device, non_blocking=True),
        [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
    )


@torch.no_grad()
def _maybe_update_duetda(train_duet, feat, batch_idx):
    if train_duet is None:
        return
    

    train_duet.score_batch(batch_feat=feat, batch_indices=batch_idx)

def _maybe_log(i, epoch, train_loader, meters, batch_time, data_time, losses):
    if i % args.print_freq != 0:
        return

    if args.task_type== "regression":
        msg = (f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
               f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
               f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
               f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
               f"MAE {meters['mae'].val:.3f} ({meters['mae'].avg:.3f})")
        log_dict = {
            "epoch": epoch, "train/step": i,
            "train/loss": float(losses.val), "train/loss_avg": float(losses.avg),
            "train/mae": float(meters["mae"].val), "train/mae_avg": float(meters["mae"].avg),
        }
    else:
        msg = (f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
               f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
               f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
               f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
               f"Accu {meters['acc'].val:.3f} ({meters['acc'].avg:.3f})\t"
               f"Precision {meters['prec'].val:.3f} ({meters['prec'].avg:.3f})\t"
               f"Recall {meters['recall'].val:.3f} ({meters['recall'].avg:.3f})\t"
               f"F1 {meters['f1'].val:.3f} ({meters['f1'].avg:.3f})\t"
               f"AUC {meters['auc'].val:.3f} ({meters['auc'].avg:.3f})")
        log_dict = {
            "epoch": epoch, "train/step": i,
            "train/loss": float(losses.val), "train/loss_avg": float(losses.avg),
            "train/accuracy": float(meters["acc"].val), "train/accuracy_avg": float(meters["acc"].avg),
            "train/precision": float(meters["prec"].val),
            "train/recall": float(meters["recall"].val),
            "train/f1": float(meters["f1"].val),
            "train/auc": float(meters["auc"].val),
        }

    print(msg)
    if args.use_wandb and wandb is not None:
        try:
            wandb.log(log_dict)
        except Exception:
            pass

        
def cgcnn_train(train_loader, model, criterion, optimizer, epoch, normalizer, train_set=None, device=torch.device("cuda"), flops_meter=None):
    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, grad_norms = AverageMeter(), AverageMeter()

    meters = {"mae": AverageMeter()} if args.task_type == "regression" else {
        "acc": AverageMeter(), "prec": AverageMeter(), "recall": AverageMeter(),
        "f1": AverageMeter(), "auc": AverageMeter()
    }

    model.train()
    end = time.time()

    for i, (batch_data, batch_idx, batch_w) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input, target, _ = batch_data
        input_var = _to_device_cgcnn_input(input, device)

        if args.task_type == "regression":
            target_var = normalizer.norm(target).to(device, non_blocking=True)
        else:
            target_var = target.view(-1).long().to(device, non_blocking=True)

        if flops_meter is not None and epoch == 0 and i == 0:
            try:
                flops_meter.profile_base_once(model, input_var)
            except Exception:
                pass

        output = model(*input_var)
        if flops_meter is not None:
            flops_meter.step_base()

        loss = criterion(output, target_var, w=None, ret_per_sample=True)
        feat = batch_data[0][-1]  # val_feat location
        feat = feat.to(device, non_blocking=True)

        if flops_meter is not None and epoch == 0 and i == 0 and hasattr(train_set, 'valuator'):
            try:
                wrap = ValuatorWrap(train_set.valuator, getattr(train_set, "gate", None)).to(device).eval()
                status = train_set.status_tracker.vector()
                if torch.is_tensor(status):
                    status = status.to(device)
                flops_meter.profile_val_once(wrap, (feat.detach(), None, None, status))
            except Exception:
                pass

        loss = train_set.update(batch_idx, loss_vec=loss, batch_feat=feat)
        if flops_meter is not None:
            flops_meter.step_val()

        bs = target.size(0)
        losses.update(loss.item(), bs)

        if args.task_type == "regression":
            mae_error = mae(normalizer.denorm(output.detach().cpu()), target)
            meters["mae"].update(mae_error, bs)
        else:
            acc, prec, recall, f1, auc = class_eval(output.detach().cpu(), target)
            meters["acc"].update(acc, bs)
            meters["prec"].update(prec, bs)
            meters["recall"].update(recall, bs)
            meters["f1"].update(f1, bs)
            meters["auc"].update(auc, bs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms.update(grad_norm_avg(model))
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
            
        _maybe_log(i, epoch, train_loader, meters, batch_time, data_time, losses)

    return {
        "avg_loss": float(losses.avg) if losses.count else float("nan"),
        "avg_grad_norm": float(grad_norms.avg) if grad_norms.count else 0.0,
    }
    
    
    

def _maybe_print_val(i, loader, batch_time, losses, meters):
    if i % args.print_freq != 0:
        return
    if args.task_type == "regression":
        print(f"Test: [{i}/{len(loader)}]\t"
              f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
              f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
              f"MAE {meters['mae'].val:.3f} ({meters['mae'].avg:.3f})")
    else:
        print(f"Test: [{i}/{len(loader)}]\t"
              f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
              f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
              f"Accu {meters['acc'].val:.3f} ({meters['acc'].avg:.3f})\t"
              f"Precision {meters['prec'].val:.3f} ({meters['prec'].avg:.3f})\t"
              f"Recall {meters['recall'].val:.3f} ({meters['recall'].avg:.3f})\t"
              f"F1 {meters['f1'].val:.3f} ({meters['f1'].avg:.3f})\t"
              f"AUC {meters['auc'].val:.3f} ({meters['auc'].avg:.3f})")

def _wandb_log(prefix, d):
    if args.use_wandb and wandb is not None:
        try:
            wandb.log({f"{prefix}/{k}": v for k, v in d.items()})
        except Exception:
            pass

def _save_test_csv(cif_ids, targets, preds, path=None):
    import csv
    dest_path = path or args.test_res_path
    dest_path = os.path.abspath(os.path.expanduser(dest_path))
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    with open(dest_path, "w") as f:
        w = csv.writer(f)
        for cid, t, p in zip(cif_ids, targets, preds):
            w.writerow((cid, t, p))




def save_checkpoint(state, is_best, filename=None, checkpoint_dir=None):
    """Persist checkpoints under the configured directory."""
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir or args.checkpoint_dir))
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        filename = f"{args.data_name}_{args.da_method}_{args.selection_ratio}_last.pth.tar"

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_filename = f"{args.data_name}_{args.da_method}_{args.selection_ratio}_model_best.pth.tar"
        best_path = os.path.join(checkpoint_dir, best_filename)
        shutil.copyfile(checkpoint_path, best_path)



def validate(val_loader, model, criterion, normalizer, test=False, val_type="val", device=torch.device("cuda")):

    batch_time, losses = AverageMeter(), AverageMeter()
    meters = {"mae": AverageMeter()} if args.task_type== "regression" else {
        "acc": AverageMeter(), "prec": AverageMeter(), "recall": AverageMeter(),
        "f1": AverageMeter(), "auc": AverageMeter()
    }

    # optional collectors
    all_targets, all_preds = [], []   # for aggregate regression metrics
    test_targets, test_preds, test_cif_ids = [], [], []

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target, batch_cif_ids) in enumerate(val_loader):
            input_var = _to_device_cgcnn_input(input, device)

            if args.task_type== "regression":
                target_var = normalizer.norm(target).to(device, non_blocking=True)
            else:
                target_var = target.view(-1).long().to(device, non_blocking=True)

            output = model(*input_var)
            loss = criterion(output, target_var)

            bs = target.size(0)
            losses.update(loss.item(), bs)

            if args.task_type== "regression":
                pred = normalizer.denorm(output.detach().cpu()).view(-1)
                tgt = target.view(-1)

                mae_val = mae(pred, tgt)
                mae_val = float(mae_val.item()) if torch.is_tensor(mae_val) else float(mae_val)
                meters["mae"].update(mae_val, bs)

                all_preds.extend(pred.tolist())
                all_targets.extend(tgt.tolist())

                if test:
                    test_preds.extend(pred.tolist())
                    test_targets.extend(tgt.tolist())
                    test_cif_ids.extend(batch_cif_ids)

            else:
                acc, prec, recall, f1, auc = class_eval(output.detach().cpu(), target)
                meters["acc"].update(acc, bs)
                meters["prec"].update(prec, bs)
                meters["recall"].update(recall, bs)
                meters["f1"].update(f1, bs)
                meters["auc"].update(auc, bs)

                if test:
                    prob = torch.exp(output.detach().cpu())
                    # assume binary, use prob of class 1
                    test_preds.extend(prob[:, 1].tolist())
                    test_targets.extend(target.view(-1).tolist())
                    test_cif_ids.extend(batch_cif_ids)

            batch_time.update(time.time() - end)
            end = time.time()
            _maybe_print_val(i, val_loader, batch_time, losses, meters)

    # --- finalize ---
    star = "**" if test else "*"
    prefix = "test" if test else f"val_{val_type}"

    if test:
        _save_test_csv(test_cif_ids, test_targets, test_preds, path=args.test_res_path)

    avg_loss_value = float(losses.avg) if losses.count else float("nan")
    val_stats = {"avg_loss": avg_loss_value, "avg_grad_norm": 0.0}

    if args.task_type== "regression":
        preds_np = np.asarray(all_preds, dtype=np.float64)
        targets_np = np.asarray(all_targets, dtype=np.float64)

        mae_value = float(meters["mae"].avg)
        if preds_np.size > 0 and preds_np.size == targets_np.size:
            rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
            std_y = float(np.std(targets_np))
            nrmse = float(rmse / (std_y + 1e-12))
            nmae = float(mae_value / (std_y + 1e-12))
            try:
                r2 = float(metrics.r2_score(targets_np, preds_np))
            except Exception:
                r2 = float("nan")
        else:
            rmse = nrmse = nmae = r2 = float("nan")

        print(f" {star} MAE {mae_value:.3f} | NMAE {nmae:.3f} | RMSE {rmse:.3f} | "
              f"NRMSE {nrmse:.3f} | R^2 {r2:.3f}")

        _wandb_log(prefix, {"mae": mae_value, "nmae": nmae, "rmse": rmse, "nrmse": nrmse, "r2": r2})

        return {"mae": mae_value, "nmae": nmae, "rmse": rmse, "nrmse": nrmse, "r2": r2}, val_stats

    else:
        print(f" {star} AUC {meters['auc'].avg:.3f}")
        _wandb_log(prefix, {"auc": float(meters["auc"].avg)})
        return float(meters["auc"].avg), val_stats

# Print concise summaries
def print_ranked(title, ranked_list, epoch_test_map):
    print(f'---------Top-10 Epochs ({title})---------------')
    for idx, entry in enumerate(ranked_list, start=1):
        e = entry['epoch']; tst = epoch_test_map.get(e)
        if args.task_type== 'regression':
            iid = entry['iid']; ood = entry['ood']
            print(f"#{idx} Epoch {e} | IID MAE {iid['mae']:.4f}, RMSE {iid['rmse']:.4f}, R2 {iid['r2']:.4f} | "
                    f"OOD MAE {ood['mae']:.4f}, RMSE {ood['rmse']:.4f}, R2 {ood['r2']:.4f} | "
                    f"Test: " + (f"MAE {tst['mae']:.4f}, RMSE {tst['rmse']:.4f}, R2 {tst['r2']:.4f}" if tst else 'N/A'))
        else:
            iid_auc = entry['iid']; ood_auc = entry['ood']
            print(f"#{idx} Epoch {e} | IID AUC {iid_auc:.4f} | OOD AUC {ood_auc:.4f} | Test AUC " + (f"{tst:.4f}" if tst is not None else 'N/A'))



def test_res_according_split(split_path="./analysis/test_id_difficulty_split.csv", results_path=None):
    final_results_path = os.path.abspath(os.path.expanduser(results_path or args.test_res_path))
    test_results = pd.read_csv(final_results_path)
    test_splits  = pd.read_csv(split_path)
    
    iid_res = {'preds': [], 'targets': []}
    ood_res = {'preds': [], 'targets': []}
    
    for i in range(len(test_results)):
        cid = test_results.iloc[i, 0][5:]  # remove 'cifs/' prefix
        pred = test_results.iloc[i, 2]
        true = test_results.iloc[i, 1]
        if cid not in test_splits['id'].values:
            print(f"Warning: CIF id {cid} not found in split info; skipping")
            continue
        difficulty = test_splits[test_splits['id'] == cid]['data_split'].values[0]
        if difficulty == 'ID':
            iid_res['preds'].append(pred)
            iid_res['targets'].append(true)
        elif difficulty == 'OOD':
            ood_res['preds'].append(pred)
            ood_res['targets'].append(true)
    ID_mae = mae(torch.tensor(iid_res['preds']), torch.tensor(iid_res['targets']))
    OOD_mae = mae(torch.tensor(ood_res['preds']), torch.tensor(ood_res['targets']))
    ID_nrmse = torch.sqrt(torch.mean((torch.tensor(iid_res['preds']) - torch.tensor(iid_res['targets'])) ** 2)) / (torch.std(torch.tensor(iid_res['targets'])) + 1e-12)
    OOD_nrmse = torch.sqrt(torch.mean((torch.tensor(ood_res['preds']) - torch.tensor(ood_res['targets'])) ** 2)) / (torch.std(torch.tensor(ood_res['targets'])) + 1e-12)
    print(f"Test ID MAE: {ID_mae:.4f}, NRMSE: {ID_nrmse:.4f} | Test OOD MAE: {OOD_mae:.4f}, NRMSE: {OOD_nrmse:.4f}")
    test_results = {
        'ID_mae': float(ID_mae),
        'OOD_mae': float(OOD_mae),
        'ID_nrmse': float(ID_nrmse),
        'OOD_nrmse': float(OOD_nrmse)
    }
    return test_results
    
    



def main():
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    # ----------- Initialize model, optimizer, criterion, data loaders, etc. -----------
    print(f"Initializing model {args.model_name}...")
    model = initialize_model(args.model_name, task_type=args.task_type)
    
    if args.cuda:
        model = model.to(device)
    print(f"Initializing dataset and dataloaders...")
    
    train_set, iid_val_set, ood_val_set, test_set  = initialize_dataset(args.data_name, args.data_root, args.fold)
     
    if args.data_name == "cgcnn_matbench":
        DataValuator = DualDataValuator(input_dim=args.da_input_dim)
        DataValuator.load_state_dict(torch.load(args.da_model_ckpt))
        if args.cuda:
            DataValuator = DataValuator.to(device)

        train_set_duet = DuetDALoader(
            dataset=train_set,
            valuator=DataValuator,
            ratio=args.selection_ratio,
            num_epoch=args.epochs,
            method=args.da_method
        )

        train_sampler = train_set_duet.pruning_sampler()

        train_loader = DataLoader(
            train_set_duet,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_pool_val_with_meta,
            drop_last=False,
            pin_memory=True,
        )

        iid_loader   = DataLoader(iid_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
        ood_loader   = DataLoader(ood_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
        test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_pool)
        

    normalizer = initialize_normalizer(train_set, args.task_type)

    flops_meter = MultiFlopsMeter(device=device)
    
    
    optimizer  = initialize_optimizer(args.optim, model, args.lr, args.momentum, args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)
    # loss_func = nn.MSELoss(reduction='none') if args.task_type == 'regression' else nn.CrossEntropyLoss(reduction='none')
    loss_func = weighted_mse_loss
    criterion = loss_func
    
    
    # ----------- Training loop -----------


    # initialize W&B if requested
    if args.use_wandb and args.wandb_mode != 'disabled':
        if wandb is None:
            print('W&B not installed. Proceeding without logging.')
            args.use_wandb = False
        else:
            print('Initializing Weights & Biases logging...')
            wandb.init(project=args.wandb_project,
                       entity=args.wandb_entity if args.wandb_entity else None,
                       name=f"{time.strftime('%Y%m%d_%H%M%S')}-{args.wandb_name}" if args.wandb_name else None,
                       config=vars(args),
                       mode=args.wandb_mode)
            try:
                wandb.watch(model, log='all')
            except Exception:
                pass
    print('---------Start Training---------------')
    print('Training set size: ', len(train_set))
    print('IID Validation set size: ', len(iid_val_set))
    print('OOD Validation set size: ', len(ood_val_set))
    print('Test set size: ', len(test_set))
    # Prepare directory for per-epoch checkpoints
    checkpoints_dir = os.path.join(args.checkpoint_dir, f'{args.model_name}_epochs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    epoch_records = []
    if args.task_type == 'regression':
        best_mae_error = 1e10
    else:
        best_mae_error = 0.
    for epoch in range(args.epochs):
        CURRENT_EPOCH = epoch
        # train for one epoch
        train_stats = cgcnn_train(train_loader, model, loss_func, optimizer, epoch, normalizer=normalizer, train_set=train_set_duet, device=device, flops_meter=flops_meter)
        avg_train_loss = train_stats['avg_loss']
        avg_grad_norm = train_stats['avg_grad_norm']
        
        # evaluate on IID validation set
        print('\n--- Evaluating on IID Validation Set ---')
        iid_metrics, iid_stats = validate(iid_loader, model, criterion, normalizer, val_type='iid',device=device)
        avg_iid_val = iid_stats['avg_loss']
        
        # evaluate on OOD validation set
        print('\n--- Evaluating on OOD Validation Set ---')
        ood_metrics, ood_stats = validate(ood_loader, model, criterion, normalizer, val_type='ood', device=device)
        avg_ood_val = ood_stats['avg_loss']
        
        train_set_duet.status_tracker.update(
            step=epoch,
            train_loss=avg_train_loss,
            iid_val_loss=avg_iid_val,
            ood_val_loss=avg_ood_val,
            meta_grad_norm=avg_grad_norm,
        )
            
        

        # log validation metrics to W&B
        log_epoch(epoch, iid_metrics, ood_metrics)
               
         # Use IID validation error for model selection
        mae_error = (iid_metrics['mae'] + ood_metrics['mae'])/2 if args.task_type== 'regression' else iid_metrics
        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint (based on IID validation)
        if args.task_type== 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

        # Also save a per-epoch checkpoint
        epoch_ckpt_path = os.path.join(checkpoints_dir, f'epoch_{epoch + 1}.pth.tar')
        try:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, epoch_ckpt_path)
        except Exception:
            epoch_ckpt_path = None
        # Record this epoch's metrics
        epoch_records.append({
            'epoch': epoch + 1,
            'iid': iid_metrics,
            'ood': ood_metrics,
            'ckpt': epoch_ckpt_path
        })

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_ckpt_name = f"{args.data_name}_{args.da_method}_{args.selection_ratio}_model_best.pth.tar"
    best_checkpoint_path = os.path.join(args.checkpoint_dir, best_ckpt_name)
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint['state_dict'])
    test_metrics, _ = validate(test_loader, model, criterion, normalizer, test=True, device=device)
    split_path = f"{args.data_root}/splits/fold{args.fold}/test_id_split.csv"
    test_res = test_res_according_split(split_path=split_path, results_path=args.test_res_path)
    
    if args.use_wandb and wandb is not None:
        if args.task_type== 'regression':
            test_log = {
                'test/mae': test_metrics['mae'],
                'test/rmse': test_metrics['rmse'],
                'test/nrmse': test_metrics['nrmse'],
                'test/nmae': test_metrics['nmae'],
                'test/r2': test_metrics['r2'],
                'test/id_mae': test_res['ID_mae'],
                'test/ood_mae': test_res['OOD_mae'],
                'test/id_nrmse': test_res['ID_nrmse'],
                'test/ood_nrmse': test_res['OOD_nrmse']
            }
            wandb.log(test_log)
        else:
            wandb.log({'test/auc': test_metrics})

    flops = flops_meter.total_flops(base_multiplier=3.0, val_multiplier=1.0)


    if args.use_wandb and wandb is not None:
        try:
            wandb.log({
                'flops/base_forward_per_step': flops['base_forward_flops'] or 0,
                'flops/valuator_forward_per_step': flops['val_forward_flops'] or 0,
                'flops/base_steps': flops['base_steps'],
                'flops/valuator_steps': flops['val_steps'],
                'flops/base_total': flops['base_total_flops'],
                'flops/valuator_total': flops['val_total_flops'],
                'flops/total_used': flops['total_flops'],
            })
        except Exception:
            pass

    
    
    # Build ranked lists: IID, OOD, and average of IID+OOD
    if args.task_type== 'regression':
        top_iid = sorted(epoch_records, key=lambda x: x['iid']['mae'])[:10]
        top_ood = sorted(epoch_records, key=lambda x: x['ood']['mae'])[:10]
        top_avg = sorted(epoch_records, key=lambda x: (x['iid']['mae'] + x['ood']['mae']) / 2.0)[:10]
    else:
        top_iid = sorted(epoch_records, key=lambda x: x['iid'], reverse=True)[:10]
        top_ood = sorted(epoch_records, key=lambda x: x['ood'], reverse=True)[:10]
        top_avg = sorted(epoch_records, key=lambda x: (x['iid'] + x['ood']) / 2.0, reverse=True)[:10]

    # Evaluate test metrics for union of top epochs across all rankings
    epochs_to_eval = {e['epoch'] for e in (top_iid + top_ood + top_avg)}
    epoch_test_map = {}
    for e in epochs_to_eval:
        rec = next((r for r in epoch_records if r['epoch'] == e), None)
        if rec and rec['ckpt'] and os.path.isfile(rec['ckpt']):
            try:
                ckpt = torch.load(rec['ckpt'])
                model.load_state_dict(ckpt['state_dict'])
                if 'normalizer' in ckpt:
                    normalizer.load_state_dict(ckpt['normalizer'])
                epoch_test_map[e], _ = validate(test_loader, model, criterion, normalizer, test=False,device=device)
            except Exception:
                epoch_test_map[e] = None
        else:
            epoch_test_map[e] = None

    print_ranked('by IID validation', top_iid, epoch_test_map)
    print_ranked('by OOD validation', top_ood, epoch_test_map)
    print_ranked('by IID+OOD average', top_avg, epoch_test_map)

    # Log W&B top-10 rankings as plain scalars (no Table)
    if args.use_wandb and wandb is not None:
        def _py(x):
            """Convert torch/numpy scalars to python scalars for wandb."""
            if x is None:
                return None
            try:
                import numpy as np
                if isinstance(x, np.generic):
                    return x.item()
            except Exception:
                pass
            if torch.is_tensor(x):
                return x.detach().cpu().item()
            if isinstance(x, (int, float)):
                return x
            try:
                return float(x)
            except Exception:
                return None

        # Log W&B top-10 rankings as plain scalars (no Table) + put into Summary
        if args.use_wandb and wandb is not None and wandb.run is not None:
            def _py(x):
                if x is None:
                    return None
                try:
                    import numpy as np
                    if isinstance(x, np.generic):
                        return x.item()
                except Exception:
                    pass
                if torch.is_tensor(x):
                    return x.detach().cpu().item()
                if isinstance(x, (int, float)):
                    return x
                try:
                    return float(x)
                except Exception:
                    return None

            def log_ranking(prefix, ranked_list):
                log_dict = {}
                for rank, it in enumerate(ranked_list, start=1):
                    e = int(it["epoch"])
                    tst = epoch_test_map.get(e, None)

                    if args.task_type== "regression":
                        iid = it["iid"]; ood = it["ood"]
                        # log keys
                        log_dict[f"{prefix}/rank_{rank}/epoch"] = e
                        log_dict[f"{prefix}/rank_{rank}/iid_mae"] = _py(iid.get("mae"))
                        log_dict[f"{prefix}/rank_{rank}/iid_rmse"] = _py(iid.get("rmse"))
                        log_dict[f"{prefix}/rank_{rank}/iid_r2"] = _py(iid.get("r2"))
                        log_dict[f"{prefix}/rank_{rank}/ood_mae"] = _py(ood.get("mae"))
                        log_dict[f"{prefix}/rank_{rank}/ood_rmse"] = _py(ood.get("rmse"))
                        log_dict[f"{prefix}/rank_{rank}/ood_r2"] = _py(ood.get("r2"))
                        log_dict[f"{prefix}/rank_{rank}/test_mae"] = _py(tst.get("mae")) if tst else None
                        log_dict[f"{prefix}/rank_{rank}/test_rmse"] = _py(tst.get("rmse")) if tst else None
                        log_dict[f"{prefix}/rank_{rank}/test_r2"] = _py(tst.get("r2")) if tst else None
                    else:
                        log_dict[f"{prefix}/rank_{rank}/epoch"] = e
                        log_dict[f"{prefix}/rank_{rank}/iid_auc"] = _py(it.get("iid"))
                        log_dict[f"{prefix}/rank_{rank}/ood_auc"] = _py(it.get("ood"))
                        log_dict[f"{prefix}/rank_{rank}/test_auc"] = _py(tst) if tst is not None else None

                # 1) normal log: do NOT set step (avoid step rollback being dropped)
                wandb.log(log_dict, commit=True)

                # 2) also write into Summary so it shows like test metrics in Overview
                for k, v in log_dict.items():
                    if isinstance(v, (int, float)) and v == v:  # numeric and not NaN
                        wandb.run.summary[k] = v

            log_ranking("top10/by_iid", top_iid)
            log_ranking("top10/by_ood", top_ood)
            log_ranking("top10/by_avg", top_avg)
            
            print(test_metrics)
            print(test_res)
            
    print('---------FLOPs Summary---------------')
    print(f"Base fwd FLOPs/step: {flops['base_forward_flops']}")
    print(f"Valuator fwd FLOPs/step: {flops['val_forward_flops']}")
    print(f"Base steps: {flops['base_steps']} | Valuator steps: {flops['val_steps']}")
    print(f"Base total FLOPs: {flops['base_total_flops']}")
    print(f"Valuator total FLOPs: {flops['val_total_flops']}")
    print(f"Total FLOPs used: {flops['total_flops']}")
    

def log_epoch(epoch, iid_metrics, ood_metrics):
    if args.use_wandb and wandb is not None:
        if args.task_type== 'regression':
            wandb.log({
                    'epoch': epoch, 
                    'val_iid/mae': iid_metrics['mae'],
                    'val_iid/rmse': iid_metrics['rmse'],
                    'val_iid/nrmse': iid_metrics['nrmse'],
                    'val_iid/nmae': iid_metrics['nmae'],
                    'val_iid/r2': iid_metrics['r2'],
                    'val_ood/mae': ood_metrics['mae'],
                    'val_ood/rmse': ood_metrics['rmse'],
                    'val_ood/nrmse': ood_metrics['nrmse'],
                    'val_ood/nmae': ood_metrics['nmae'],
                    'val_ood/r2': ood_metrics['r2'],
                    'val/gap_nrmse_ood_minus_iid': ood_metrics['nrmse'] - iid_metrics['nrmse'],
                    'val/gap_nmae_ood_minus_iid': ood_metrics['nmae'] - iid_metrics['nmae'],
                })
        else:
            wandb.log({
                    'epoch': epoch, 
                    'val_iid/auc': iid_metrics,
                    'val_ood/auc': ood_metrics,
                })
        
    
    
    

if __name__ == "__main__":
    main()
    