import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cgcnn.cgcnn.data import collate_pool_val_with_meta
from modules.data_select import DuetDALoader


def _to_device_cgcnn_input(input, device):
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = input[:4]
    return (
        atom_fea.to(device, non_blocking=True),
        nbr_fea.to(device, non_blocking=True),
        nbr_fea_idx.to(device, non_blocking=True),
        [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
    )



class _SimpleNormalizer:
    def __init__(self, x: torch.Tensor, eps: float = 1e-12):
        x = x.float().view(-1)
        self.mean = x.mean()
        self.std = x.std().clamp_min(eps)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def _find_last_linear(model: nn.Module) -> nn.Linear:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("No nn.Linear found in model (needed for GraNd/Influence proxy).")
    return last


def _collect_targets(train_loader, task_type: str, device):
    ys = []
    for (batch_data, _, _) in train_loader:
        _, target, _ = batch_data
        if task_type == "regression":
            ys.append(target.view(-1).float())
        else:
            ys.append(target.view(-1).long().float())  # not used, but keep shape
    return torch.cat(ys, dim=0)


def _proxy_train_epochs(model, train_loader, task_type: str, normalizer, device,
                        epochs: int, lr: float, weight_decay: float,
                        forget_thresh: float, track_forgetting: bool):
    model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(train_loader.dataset)  # this is DuetDALoader wrapping train_set
    prev_good = np.zeros(n, dtype=np.bool_)
    forgetting = np.zeros(n, dtype=np.int32)

    for ep in range(epochs):
        for (batch_data, batch_idx, _) in train_loader:
            inp, target, _ = batch_data
            inp_var = _to_device_cgcnn_input(inp, device)

            if task_type == "regression":
                y = normalizer.norm(target.view(-1).float()).to(device)
                out = model(*inp_var).view(-1)
                per_loss = (out - y).pow(2)  # per-sample MSE
                loss = per_loss.mean()
                good = (per_loss.detach().cpu().numpy() < forget_thresh)
            else:
                y = target.view(-1).long().to(device)
                logits = model(*inp_var)
                per_loss = F.cross_entropy(logits, y, reduction="none")
                loss = per_loss.mean()
                pred = logits.detach().argmax(dim=-1).cpu().numpy()
                good = (pred == y.detach().cpu().numpy())

            if track_forgetting:
                idx_np = batch_idx.detach().cpu().numpy().astype(np.int64)
                forgetting[idx_np] += (prev_good[idx_np] & (~good)).astype(np.int32)
                prev_good[idx_np] = good

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return forgetting.astype(np.float32) if track_forgetting else None


@torch.no_grad()
def _score_el2n(model, loader, task_type: str, normalizer, device):
    model.eval()
    n = len(loader.dataset)
    scores = np.zeros(n, dtype=np.float32)

    for (batch_data, batch_idx, _) in loader:
        inp, target, _ = batch_data
        inp_var = _to_device_cgcnn_input(inp, device)

        if task_type == "regression":
            y = target.view(-1).float().to(device)
            out = model(*inp_var).view(-1)
            out = normalizer.denorm(out)
            s = (out - y).abs()
        else:
            y = target.view(-1).long().to(device)
            logits = model(*inp_var)
            p = torch.softmax(logits, dim=-1)
            onehot = torch.zeros_like(p).scatter_(1, y.view(-1, 1), 1.0)
            s = torch.norm(p - onehot, dim=1)

        idx = batch_idx.detach().cpu().numpy().astype(np.int64)
        scores[idx] = s.detach().cpu().numpy().astype(np.float32)

    return scores


@torch.no_grad()
def _score_grand(model, loader, task_type: str, normalizer, device):
    """
    GraNd proxy: ||dL/dlogits|| * ||x|| where x is input to last linear layer.
    (Matches common efficient implementation used in practice for last-layer grad norm.)
    """
    model.eval()
    n = len(loader.dataset)
    scores = np.zeros(n, dtype=np.float32)

    last_fc = _find_last_linear(model)
    cache = {"x": None}

    def _hook(mod, inputs, outputs):
        cache["x"] = inputs[0].detach()  # [B, D]

    h = last_fc.register_forward_hook(_hook)

    for (batch_data, batch_idx, _) in loader:
        inp, target, _ = batch_data
        inp_var = _to_device_cgcnn_input(inp, device)

        if task_type == "regression":
            y = normalizer.norm(target.view(-1).float()).to(device)
            out = model(*inp_var).view(-1)             # logits-like (scalar)
            x = cache["x"]                             # [B, D]
            dl = (2.0 * (out - y)).abs()               # ||dL/dy|| since out_dim=1
            s = dl * torch.norm(x, dim=1)
        else:
            y = target.view(-1).long().to(device)
            logits = model(*inp_var)                   # [B, C]
            x = cache["x"]                             # [B, D]
            p = torch.softmax(logits, dim=-1)
            onehot = torch.zeros_like(p).scatter_(1, y.view(-1, 1), 1.0)
            dl = p - onehot                            # dCE/dlogits
            s = torch.norm(dl, dim=1) * torch.norm(x, dim=1)

        idx = batch_idx.detach().cpu().numpy().astype(np.int64)
        scores[idx] = s.detach().cpu().numpy().astype(np.float32)

    h.remove()
    return scores


def _val_last_layer_grad(model, loader, task_type: str, normalizer, device):
    """
    Compute gradient of mean val loss w.r.t last linear layer weights: g_val (C,D) or (1,D).
    """
    model.train(False)
    last_fc = _find_last_linear(model)
    cache = {"x": None}

    def _hook(mod, inputs, outputs):
        cache["x"] = inputs[0]  # keep grad graph

    h = last_fc.register_forward_hook(_hook)
    model.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0

    for (batch_data, _, _) in loader:
        inp, target, _ = batch_data
        inp_var = _to_device_cgcnn_input(inp, device)

        if task_type == "regression":
            y = normalizer.norm(target.view(-1).float()).to(device)
            out = model(*inp_var).view(-1)
            loss = (out - y).pow(2).mean()
        else:
            y = target.view(-1).long().to(device)
            logits = model(*inp_var)
            loss = F.cross_entropy(logits, y, reduction="mean")

        bs = target.size(0)
        total_loss = total_loss + loss * bs
        total_n += bs

    (total_loss / max(total_n, 1)).backward()
    g = last_fc.weight.grad.detach().clone()  # [C, D] or [1, D]
    h.remove()
    model.zero_grad(set_to_none=True)
    return g


@torch.no_grad()
def _score_grad_align(model, loader, task_type: str, normalizer, device, g_val, n_local, g2l=None):
    """
    Influence/DP cheap proxy:
      score_i = < g_val , g_i(last-layer) >
    支持 batch_idx 既可能是 local idx，也可能是 global idx（需要 g2l 映射）
    """
    def _to_local_idx(batch_idx):
        idx_raw = batch_idx.detach().cpu().numpy().astype(np.int64)
        # 如果 idx_raw 已经是 local（都 < n_local），直接用
        if idx_raw.size == 0 or (idx_raw.max() < n_local and idx_raw.min() >= 0):
            return idx_raw
        # 否则当作 global idx，需要映射
        if g2l is None:
            raise RuntimeError(
                f"batch_idx looks global (max={idx_raw.max()}) but no g2l map provided. "
                "Pass g2l built from train_set.indices."
            )
        return np.asarray([g2l[int(x)] for x in idx_raw], dtype=np.int64)

    model.eval()
    scores = np.zeros(n_local, dtype=np.float32)

    last_fc = _find_last_linear(model)
    cache = {"x": None}

    def _hook(mod, inputs, outputs):
        cache["x"] = inputs[0].detach()  # [B, D]

    h = last_fc.register_forward_hook(_hook)

    g_val = g_val.to(device)              # [C, D] or [1, D]
    g_val_T = g_val.t().contiguous()      # [D, C] or [D,1]

    for (batch_data, batch_idx, _) in loader:
        inp, target, _ = batch_data
        inp_var = _to_device_cgcnn_input(inp, device)

        if task_type == "regression":
            y = normalizer.norm(target.view(-1).float()).to(device)
            out = model(*inp_var).view(-1)                 # [B]
            x = cache["x"]                                 # [B, D]
            dl = (2.0 * (out - y)).view(-1, 1)            # [B,1]
            proj = (x @ g_val_T).view(-1, 1)              # [B,1]
            s = (dl * proj).view(-1)                      # [B]
        else:
            y = target.view(-1).long().to(device)
            logits = model(*inp_var)                       # [B, C]
            x = cache["x"]                                 # [B, D]
            p = torch.softmax(logits, dim=-1)
            onehot = torch.zeros_like(p).scatter_(1, y.view(-1, 1), 1.0)
            dl = (p - onehot)                              # [B, C]
            proj = x @ g_val_T                             # [B, C]
            s = (dl * proj).sum(dim=1)                     # [B]

        idx = _to_local_idx(batch_idx)                     # ✅ 关键：映射到 local
        scores[idx] = s.detach().cpu().numpy().astype(np.float32)

    h.remove()
    return scores

def _build_g2l(train_set):
    if hasattr(train_set, "indices"):
        return {int(g): i for i, g in enumerate(train_set.indices)}
    return None

def get_train_sub_set_static(proxy_model, train_set, selection_ratio, da_method, device, args=None):
    """
    Run proxy training for several epochs and return pruned Subset(train_set, keep_idx).
    Supported da_method: el2n, grand, forgetting, influence, dp
    """
    g2l = _build_g2l(train_set)
    n_local = len(train_set)
    # ---- config (from args if present)
    proxy_epochs = int(getattr(args, "proxy_epochs", 3))
    lr = float(getattr(args, "proxy_lr", getattr(args, "lr", 1e-3)))
    wd = float(getattr(args, "proxy_wd", getattr(args, "weight_decay", 0.0)))
    forget_thresh = float(getattr(args, "forget_thresh", 0.05))  # for regression "good/bad"
    val_ratio = float(getattr(args, "proxy_val_ratio", 0.1))     # for influence/dp proxy

    # ---- wrap dataset to get (data, idx, w)
    tmp_ds = DuetDALoader(dataset=train_set, valuator=None, ratio=1.0, num_epoch=None, method="topk")
    train_loader = DataLoader(
        tmp_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate_pool_val_with_meta, drop_last=False, pin_memory=True
    )
    full_loader = DataLoader(
        tmp_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=collate_pool_val_with_meta, drop_last=False, pin_memory=True
    )

    # ---- normalizer for regression
    if args.task_type == "regression":
        y_all = _collect_targets(full_loader, args.task_type, device)
        normalizer = _SimpleNormalizer(y_all)
    else:
        normalizer = None

    # ---- train proxy & (optionally) collect forgetting stats
    track_forgetting = (da_method == "forgetting")
    forgetting_scores = _proxy_train_epochs(
        proxy_model, train_loader, args.task_type, normalizer, device,
        epochs=proxy_epochs, lr=lr, weight_decay=wd,
        forget_thresh=forget_thresh, track_forgetting=track_forgetting
    )

    # ---- compute static scores
    if da_method == "forgetting":
        scores = forgetting_scores

    elif da_method == "el2n":
        scores = _score_el2n(proxy_model, full_loader, args.task_type, normalizer, device)

    elif da_method == "grand":
        scores = _score_grand(proxy_model, full_loader, args.task_type, normalizer, device)

    elif da_method in ["influence", "dp"]:
        # cheap influence/DP proxy needs a "val" objective; use a heldout slice from train_set
        n = len(tmp_ds)
        rng = np.random.default_rng(int(getattr(args, "seed", 0)))
        perm = rng.permutation(n)
        nv = max(1, int(val_ratio * n))
        val_idx = perm[:nv].tolist()
        tr_idx = perm[nv:].tolist()

        # build loaders on the wrapped tmp_ds indices (note: Subset over tmp_ds)
        val_ds = Subset(tmp_ds, val_idx)
        tr_ds  = Subset(tmp_ds, tr_idx)

        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=collate_pool_val_with_meta, drop_last=False, pin_memory=True
        )
        tr_loader = DataLoader(
            tr_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            collate_fn=collate_pool_val_with_meta, drop_last=False, pin_memory=True
        )

        # compute g_val on last layer
        g_val = _val_last_layer_grad(proxy_model, val_loader, args.task_type, normalizer, device)

        # score training examples by gradient alignment
        scores = _score_grad_align(
            proxy_model, full_loader, args.task_type, normalizer, device,
            g_val=g_val, n_local=n_local, g2l=g2l
        )

        # scores_sub is sized len(tr_ds) but indexed by original idx in tmp_ds
        # rebuild full scores
        scores = np.full(n, -1e9, dtype=np.float32)
        # map back: tr_loader returns batch_idx from tmp_ds global indexing
        for (batch_data, batch_idx, _) in tr_loader:
            # recompute alignment for this batch (cheaper than storing mapping arrays)
            pass
        # Instead: compute directly on full_loader using g_val (simpler & deterministic)
        scores = _score_grad_align(
            proxy_model, full_loader, args.task_type, normalizer, device,
            g_val=g_val, n_local=n_local, g2l=g2l
        )

    else:
        raise ValueError(f"Unknown da_method: {da_method}")

    # ---- prune top-k
    n = len(train_set)
    k = max(1, int(selection_ratio * n))
    keep = np.argsort(-scores)[:k].astype(np.int64).tolist()

    return Subset(train_set, keep)
