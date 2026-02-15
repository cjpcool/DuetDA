import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch.nn as nn
import sys
import os

from modules.utils import EMAGeneralizationStatus, ema_beta_schedule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from data_val import DualDataValuator
from abc import abstractmethod

from torch.utils.data import Dataset,Sampler
import numpy as np



__all__ = ["DuetDALoader", "DuetDASampler"]


class BaseScorer:
    """Update per-sample scores given batch info."""
    def begin_epoch(self, epoch: int): pass
    @torch.no_grad()
    def update(self, *, idx, y=None, yhat=None, logits=None, loss_vec=None,
               grad_last=None, correct=None, val_grad_last=None):
        raise NotImplementedError




class DuetDALoader(Dataset):
    """
    MolPeg-style dataset wrapper:
      - returns (data, index, weight)
      - maintains global per-sample scores
      - provides a sampler that prunes/keeps top-ratio each epoch
    """
    def __init__(
        self,
        dataset: Dataset,
        valuator,                # DualDataValuator or any callable: score = valuator(feat, ...)
        ratio: float = 0.5,      # keep ratio
        num_epoch: int | None = None,
        method: str = "duetda",    # duetda| "random" (easy to extend)
        scores: np.ndarray | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.valuator = valuator
        if self.valuator is not None:
            self.valuator.eval()
        self.ratio = float(ratio)
        self.num_epoch = num_epoch
        self.method = method

        n = len(dataset)
        self.scores = np.ones(n, dtype=np.float32) * 3 if scores is None else scores.astype(np.float32)
        self.weights = np.ones(n, dtype=np.float32)  # importance weights after pruning
        self.all_losses = np.ones(n, dtype=np.float32) * 3  # optional: track all losses
        # self.all_losses = None
        # keep behavior consistent with common dataset wrappers
        self.transform = getattr(dataset, "transform", None)
        self.save_num = 0
        
        self.status_tracker = None
        if self.valuator is not None and hasattr(self.valuator, "gate"):
            self.status_tracker = EMAGeneralizationStatus(
                beta_schedule=ema_beta_schedule(steps=num_epoch, max_beta=0.9),
                expected_dim=self.valuator.gate.status_dim,
                use_deltas=True,  # set False if you only want level terms
                device=next(self.valuator.parameters()).device,
                normalize=True,
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return data, int(index), float(self.weights[index])

    @torch.no_grad()
    def valuator_forward_only(self, batch_feat: torch.Tensor | None = None, weight_phy=None, weight_gen=None, status=None, **kwargs) -> torch.Tensor:
        """
        Compute DuetDA combined scores for a batch without updating self.scores.
        Returns: scores tensor on the same device as batch_feat.
        """
        if self.valuator is None:
            raise ValueError("valuator is not set for this DuetDALoader.")

        return  self.valuator(batch_feat, weight_phy=weight_phy, weight_gen=weight_gen, status=status)
    
    
    def update(
        self,
        batch_indices,                # [B] sample indices (int list/np/tensor)
        batch_feat: torch.Tensor | None = None,     # [B, ...] features used by valuator
        loss_vec: torch.Tensor | None = None,  # per-sample loss [B] if needed
        weight_phy=None,
        weight_gen=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute DuetDA combined scores for a batch and write back to self.scores.
        Returns: scores tensor on the same device as batch_feat.
        """
        if isinstance(batch_indices, torch.Tensor):
            idx = batch_indices.detach().cpu().numpy().astype(np.int64)
        else:
            idx = np.asarray(batch_indices, dtype=np.int64)
        
        # valuator is expected to output [B] or [B,1]
        
        if self.method == "duetda":
            if loss_vec is not None and hasattr(self, "all_losses") and self.all_losses is not None:
                self.all_losses[idx] = loss_vec.detach().float().cpu().numpy()
            status = self.status_tracker.vector() if hasattr(self, "status_tracker") else None
            s = self.valuator_forward_only(batch_feat, weight_phy=weight_phy, weight_gen=weight_gen, status=status)
            s = s.view(-1)
        else:
            # random scores for "random" method
            B = batch_feat.size(0)
            s = torch.rand(B, device=batch_feat.device, dtype=batch_feat.dtype)
        
        self.scores[idx] = s.detach().float().cpu().numpy()
        
        if loss_vec is not None:
            # rescale loss accroding to weights
            loss_rescaled = (loss_vec * torch.tensor(self.weights[idx], device=loss_vec.device, dtype=loss_vec.dtype)).mean()
        
        return loss_rescaled if loss_vec is not None else s

    def set_scores(self, indices, values):
        """Manual score update (MolPeg __setscore__ equivalent)."""
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.scores[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float32)

    def reset_weights(self):
        self.weights[:] = 1.0

    def prune(self, seed: int):
        """
        Return kept indices for this epoch.
        Also sets importance weights: kept samples get weight = 1/ratio.
        """
        n = len(self)
        k = max(1, int(self.ratio * n))
        rng = np.random.default_rng(seed)

        if self.all_losses is None:
            well_learned_indices = []
            remained_indices = []
        else:
            well_learned_mask = self.all_losses < self.all_losses.mean()
            well_learned_indices = np.where(well_learned_mask)[0]
            remained_indices = np.where(~well_learned_mask)[0].tolist()
        
        if self.method == "duetda":
            # self.scores[well_learned_indices] = -np.inf  # exclude well-learned samples
            # selected_indices = np.argsort(-self.scores)[:k]
            num_selected = len(remained_indices)
            num_needed = k - num_selected if num_selected < k else 0
            self.scores[remained_indices] = -np.inf  # exclude remained samples
            selected_indices = np.argsort(-self.scores)[:num_needed]
        else:
            selected_indices = rng.choice(n, size=k, replace=False)
        self.reset_weights()
        if selected_indices.size > 0:
            self.weights[selected_indices] = 1.0 / max(self.ratio, 1e-12)

            remained_indices.extend(selected_indices)
        np.random.shuffle(remained_indices)
        ###### ensure each epoch prune same sample numbers
        remained_indices = remained_indices[:k]
        self.save_num += int(selected_indices.size)
        
        return remained_indices

    def no_prune(self, seed: int):
        rng = np.random.default_rng(seed)
        keep = np.arange(len(self), dtype=np.int64)
        rng.shuffle(keep)
        return keep

    def pruning_sampler(self, delta: float = 1.0):
        return DuetDASampler(self, delta=delta)

    def mean_score(self):
        return float(self.scores.mean())

    def total_save(self):
        return int(self.save_num)
    

class DuetDASampler():
    def __init__(self, dataset, delta: float = 1.0):
        
        self.dataset = dataset
        self.delta = delta
        self.stop_prune = dataset.num_epoch * delta if dataset.num_epoch is not None else sys.maxsize
        self.iterations = 1
        self.iter_obj = None
        self.sample_indices = list(range(len(self.dataset)))
    
    def __getitem__(self, idx):
        return self.sample_indices[idx]
    
    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune(self.iterations)
        else:
            self.sample_indices = self.dataset.prune(self.iterations)

        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1
    
    def __next__(self):
        return next(self.iter_obj) # may raise StopIteration
        
    def __len__(self):
        return len(self.sample_indices)
    
    def __iter__(self):
        self.reset()
        return iter(self.sample_indices)
    


# class DuetDASampler(Sampler):
#     """
#     Epoch-wise pruning sampler (finite).
#     Each epoch (i.e., each __iter__ call):
#       - before stop_prune: uses dataset.prune(seed)
#       - after stop_prune: uses dataset.no_prune(seed) and resets weights once
#     """
#     def __init__(self, duetda_dataset, num_epoch=None, delta: float = 1.0):
#         self.ds = duetda_dataset
#         self.stop_prune = num_epoch if num_epoch is not None else sys.maxsize

#         self.seed = 0
#         self.seq = None

#     def reset(self):
#         self.seed += 1
#         if self.seed > self.stop_prune:
#             if self.seed == self.stop_prune + 1:
#                 self.ds.reset_weights()
#             self.seq = self.ds.no_prune(self.seed)
#         else:
#             self.seq = self.ds.prune(self.seed)

#         return self.seq

#     def __iter__(self):
#         # 每个 epoch DataLoader 都会调用一次 __iter__，
#         # 这里生成一个“有限”的 index 序列，然后返回它的 iterator
#         seq = self.reset()
#         return iter(seq.tolist() if hasattr(seq, "tolist") else seq)

#     def __len__(self):
#         if self.seq is None:
#             n = len(self.ds)
#             k = max(1, int(self.ds.ratio * n))
#             return k
#         return len(self.seq)




class UCBLoader(DuetDALoader):
    def __init__(
        self,
        dataset: Dataset,
        valuator,                # DualDataValuator or any callable: score = valuator(feat, ...)
        ratio: float = 0.5,      # keep ratio
        num_epoch: int | None = None,
        method: str = "ucb",    # duetda| "random" (easy to extend)
        scores: np.ndarray | None = None,
    ):
        super().__init__(dataset, valuator, ratio, num_epoch, method, scores)
        
        # --- bandit stats for UCB / eps-greedy ---
        n = len(dataset)
        self.n_seen = np.zeros(n, dtype=np.int64)
        self.mean_reward = np.zeros(n, dtype=np.float32)     # online mean of reward
        self.reward_ema = np.zeros(n, dtype=np.float32)      # optional smoother
        self.reward_ema_beta = 0.9                           # optional
        
        
    def update(
        self,
        batch_indices,                # [B] sample indices (int list/np/tensor)
        loss_vec: torch.Tensor | None = None,  # per-sample loss [B] if reward_mode=="neg_loss"
        use_ema: bool = False,        # whether to keep reward_ema updated (optional)
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute DuetDA combined scores for a batch and write back to self.scores.
        Returns: scores tensor on the same device as batch_feat.
        """
        if isinstance(batch_indices, torch.Tensor):
            idx = batch_indices.detach().cpu().numpy().astype(np.int64)
        else:
            idx = np.asarray(batch_indices, dtype=np.int64)
        
        # ---- choose reward for bandit update
        if loss_vec is None:
            raise ValueError("reward_mode='neg_loss' requires loss_vec (per-sample loss).")
        r_t = (-loss_vec.view(-1)).clone().detach().float()      # higher reward = smaller loss
        
        r = r_t.cpu().numpy().astype(np.float32)             # [B]

        # ---- online mean update: mean <- mean + (r-mean)/n
        # update count first
        self.n_seen[idx] += 1
        n = self.n_seen[idx].astype(np.float32)              # [B]
        mu = self.mean_reward[idx]                           # [B]
        mu = mu + (r - mu) / np.maximum(n, 1.0)
        self.mean_reward[idx] = mu

        # ---- optional EMA smoother
        if use_ema:
            beta = float(getattr(self, "reward_ema_beta", 0.9))
            self.reward_ema[idx] = beta * self.reward_ema[idx] + (1.0 - beta) * r
            
        # reweight loss_vec accordingly
        if loss_vec is not None:
            loss_vec = loss_vec * torch.tensor(self.weights[idx], device=loss_vec.device, dtype=loss_vec.dtype)
        return loss_vec.mean()


    def prune(self, seed: int):
        n = len(self)
        k = max(1, int(self.ratio * n))
        rng = np.random.default_rng(seed)

        if self.method == "ucb":
            t = int(self.n_seen.sum()) + 1
            c = getattr(self, "ucb_c", 1.0)
            bonus = c * np.sqrt(np.log(t) / (self.n_seen + 1.0))
            score = self.mean_reward + bonus
            keep = np.argsort(-score)[:k]

        elif self.method == "eps_greedy":
            eps = getattr(self, "eps", 0.1)
            if rng.random() < eps:
                keep = rng.choice(n, size=k, replace=False)
            else:
                keep = np.argsort(-self.mean_reward)[:k]

        self.weights[keep] = 1.0 / max(self.ratio, 1e-12)
        rng.shuffle(keep)
        return keep







# -----------------------------------------------------------------------------------  Selector -----------------------------------------------------------------------------------
class DataSelector(nn.Module):
    def __init__(self, valuator: DualDataValuator, selection_ratio: float):
        super(DataSelector, self).__init__()

        self.valuator = valuator
        self.selection_ratio = selection_ratio

    @abstractmethod
    def get_feature(self, batch_data, **kwargs):
        """
        Extract features from batch data for valuation.
        This is a placeholder function and should be implemented based on actual data structure.
        """
        # Assuming batch_data is already in feature form for simplicity
        raise NotImplementedError("get_feature method needs to be implemented based on data structure.")
    
    def get_data(self, batch_data, select_idx):
        """
        Get selected data from batch data based on selected indices.
        Supports different data types: dict, list, tuple, or tensor.
        """
        if isinstance(batch_data, dict):
            # Handle dictionary batch data
            selected_data = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    selected_data[key] = value[select_idx]
                elif isinstance(value, list):
                    selected_data[key] = [value[i] for i in select_idx]
                else:
                    selected_data[key] = value
            return selected_data
        elif isinstance(batch_data, (list, tuple)):
            # Handle list/tuple batch data
            selected_data = []
            for item in batch_data:
                if isinstance(item, torch.Tensor):
                    selected_data.append(item[select_idx])
                elif isinstance(item, list):
                    selected_data.append([item[i] for i in select_idx])
                else:
                    selected_data.append(item)
            return type(batch_data)(selected_data)
        else:
            # Handle tensor or other types
            return batch_data[select_idx]
    
    def forward(self, batch_data, weight_phy=None, weight_gen=None, status=None, mode='selection', **kwargs):
        """
        valuate batch data via dualdata valuator with physics score and gen score, then select a portion of data in this batch.
        """
        # feature = self.get_feature(batch_data, **kwargs)
        feature  = batch_data[0][-1]
        
        combined_scores = self.valuator(feature, weight_phy=weight_phy, weight_gen=weight_gen, status=status)

        # Determine number of samples to select
        if mode=='selection':
            num_samples = feature.size(0)
            num_select = max(1, int(num_samples * self.selection_ratio))

            # Get indices of top combined scores
            _, selected_indices = torch.topk(combined_scores, num_select, dim=0)
            # selected_indices = torch.randperm(num_samples)[:num_select]
            # Select data
            ret = self.get_data(batch_data, selected_indices.squeeze())
        elif mode=='valuation':
            ret = combined_scores
        return ret


class DataSelectorCGCNN(DataSelector):
    def __init__(self, valuator: DualDataValuator, selection_ratio: float):
        super(DataSelectorCGCNN, self).__init__(valuator, selection_ratio)

    def get_data(self, batch_data, select_idx):
        """
        Faster subset selection for CGCNN collated batch produced by collate_pool.
        Keeps the same output structure.
        """
        (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx, val_feat), batch_target, batch_cif_ids = batch_data
        device = batch_atom_fea.device

        # ---- normalize select_idx -> List[int]
        if isinstance(select_idx, torch.Tensor):
            if select_idx.dtype == torch.bool:
                select_idx = torch.nonzero(select_idx, as_tuple=False).view(-1)
            select_idx = select_idx.detach().cpu().tolist()
        else:
            select_idx = list(select_idx)

        if len(select_idx) == 0:
            empty_atom = batch_atom_fea[:0]
            empty_nbr_fea = batch_nbr_fea[:0]
            empty_nbr_idx = batch_nbr_fea_idx[:0]
            empty_target = batch_target[:0]
            return (empty_atom, empty_nbr_fea, empty_nbr_idx, []), empty_target, []

        # ---- collect per-crystal (old_start, n_i)
        # crystal_atom_idx[i] is contiguous arange + base_idx by your collate_pool
        old_starts = []
        lens = []
        for i in select_idx:
            idxs = crystal_atom_idx[i]
            # idxs may be on CPU; only need scalars here
            old_start = int(idxs[0].item()) if idxs.numel() > 0 else 0
            n_i = int(idxs.numel())
            old_starts.append(old_start)
            lens.append(n_i)

        total_atoms = sum(lens)
        if total_atoms == 0:
            empty_atom = batch_atom_fea[:0]
            empty_nbr_fea = batch_nbr_fea[:0]
            empty_nbr_idx = batch_nbr_fea_idx[:0]
            new_target = batch_target[select_idx]
            new_cif_ids = [batch_cif_ids[i] for i in select_idx]
            new_crystal_atom_idx = [torch.empty(0, dtype=torch.long, device=device) for _ in select_idx]
            return (empty_atom, empty_nbr_fea, empty_nbr_idx, new_crystal_atom_idx), new_target, new_cif_ids

        # ---- preallocate output tensors
        new_atom_fea = torch.empty((total_atoms,) + batch_atom_fea.shape[1:], device=device, dtype=batch_atom_fea.dtype)
        new_nbr_fea  = torch.empty((total_atoms,) + batch_nbr_fea.shape[1:],  device=device, dtype=batch_nbr_fea.dtype)
        new_nbr_idx  = torch.empty((total_atoms,) + batch_nbr_fea_idx.shape[1:], device=device, dtype=batch_nbr_fea_idx.dtype)

        new_crystal_atom_idx = []
        write_ptr = 0

        for old_start, n_i in zip(old_starts, lens):
            if n_i == 0:
                new_crystal_atom_idx.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            new_start = write_ptr
            delta = new_start - old_start  # shift for indices

            # slice copy (contiguous)
            new_atom_fea[new_start:new_start+n_i] = batch_atom_fea[old_start:old_start+n_i]
            new_nbr_fea[new_start:new_start+n_i]  = batch_nbr_fea[old_start:old_start+n_i]

            # neighbor idx shift (within-crystal indices stay valid)
            old_idx_block = batch_nbr_fea_idx[old_start:old_start+n_i]
            new_nbr_idx[new_start:new_start+n_i] = old_idx_block + delta

            new_crystal_atom_idx.append(
                torch.arange(new_start, new_start + n_i, device=device, dtype=torch.long)
            )
            write_ptr += n_i

        new_target = batch_target[select_idx]
        new_cif_ids = [batch_cif_ids[i] for i in select_idx]

        return (new_atom_fea, new_nbr_fea, new_nbr_idx, new_crystal_atom_idx), new_target, new_cif_ids

    def get_feature(self, batch_data, max_n=10, normalize_idx=True):
        """
        Faster feature extraction exploiting contiguity from collate_pool.
        Output: [B, D], where D = max_n * (Fa + M*Fn + M + 1)
        """
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, val_feat = batch_data[0]
        device = atom_fea.device

        B = len(crystal_atom_idx)
        Fa = atom_fea.size(1)
        M  = nbr_fea.size(1)
        Fn = nbr_fea.size(2)

        # flatten neighbor features once: [N, M, Fn] -> [N, M*Fn]
        nbr_fea_flat = nbr_fea.flatten(1)  # view/reshape, cheap

        # preallocate once for the whole batch
        atom_pad = atom_fea.new_zeros((B, max_n, Fa))                  # [B, max_n, Fa]
        nbr_pad  = nbr_fea_flat.new_zeros((B, max_n, M * Fn))          # [B, max_n, M*Fn]
        idx_pad  = nbr_fea_idx.new_zeros((B, max_n, M))                # [B, max_n, M] (long)
        mask     = atom_fea.new_zeros((B, max_n, 1))                   # [B, max_n, 1]

        for b, idxs in enumerate(crystal_atom_idx):
            n_i = int(idxs.numel())
            if n_i == 0:
                continue

            # contiguous property: base is the first global atom index
            base = int(idxs[0].item())
            n_use = min(n_i, max_n)

            # slice copy (fast)
            s = slice(base, base + n_use)
            atom_pad[b, :n_use] = atom_fea[s]
            nbr_pad[b, :n_use]  = nbr_fea_flat[s]

            nb_local = nbr_fea_idx[s] - base  # [n_use, M], local indices

            # if truncated, neighbors may point outside [0, n_use-1] -> set to 0
            nb_local = nb_local.clamp_min(0)
            nb_local[nb_local >= n_use] = 0
            idx_pad[b, :n_use] = nb_local

            mask[b, :n_use, 0] = 1.0

        # neighbor idx as numeric features (or you can switch to embedding, see note below)
        if normalize_idx:
            idx_feat = idx_pad.float().div_(float(max_n))  # [B, max_n, M]
        else:
            idx_feat = idx_pad.float()

        node_feat = torch.cat([atom_pad, nbr_pad, idx_feat, mask], dim=2)  # [B, max_n, ...]
        return node_feat.reshape(B, -1)