import torch
import torch.nn as nn
from typing import List, Tuple


class ALIGNNLayer(nn.Module):
    """Lightweight ALIGNN-style block operating on node and edge states."""

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(node_dim, edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, edge_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        if edge_feat.numel() > 0:
            edge_input = torch.cat([edge_feat, node_feat[src], node_feat[dst]], dim=-1)
            edge_feat = edge_feat + self.edge_mlp(edge_input)
        agg = node_feat.new_zeros(node_feat.size(0), edge_feat.size(1))
        if dst.numel() > 0:
            agg.index_add_(0, dst, edge_feat)
        counts = node_feat.new_zeros(node_feat.size(0), 1)
        if dst.numel() > 0:
            ones = node_feat.new_ones((dst.size(0), 1))
            counts.index_add_(0, dst, ones)
        agg = agg / counts.clamp_min(1.0)
        node_input = torch.cat([node_feat, agg], dim=-1)
        node_feat = node_feat + self.node_mlp(node_input)
        return node_feat, edge_feat


class CrystalGraphALIGNN(nn.Module):
    """ALIGNN-inspired network compatible with CGCNN-style batches."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        node_dim: int = 128,
        edge_dim: int = 64,
        readout_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification
        self.edge_threshold = 1e-6
        self.atom_proj = nn.Linear(orig_atom_fea_len, node_dim)
        self.edge_proj = nn.Linear(nbr_fea_len, edge_dim)
        self.layers = nn.ModuleList(
            [ALIGNNLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(node_dim, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        if classification:
            self.fc_out = nn.Linear(readout_dim, 2)
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(readout_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _build_graph(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, M = nbr_fea_idx.shape
        device = atom_fea.device
        src = torch.arange(N, device=device).unsqueeze(1).expand(N, M).reshape(-1)
        dst = nbr_fea_idx.reshape(-1).clamp_min(0).clamp_max(N - 1)
        edge_attr = nbr_fea.reshape(-1, nbr_fea.size(-1))
        mask = edge_attr.abs().sum(dim=1) > self.edge_threshold
        if mask.any():
            src = src[mask]
            dst = dst[mask]
            edge_attr = edge_attr[mask]
        else:
            src = src.new_empty(0)
            dst = dst.new_empty(0)
            edge_attr = edge_attr.new_empty((0, edge_attr.size(-1)))
        if src.numel() > 0:
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = torch.empty(2, 0, device=device, dtype=torch.long)
        return edge_index, edge_attr

    def pooling(self, atom_fea: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        pooled = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(pooled, dim=0)

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: List[torch.Tensor],
    ) -> torch.Tensor:
        node_feat = self.atom_proj(atom_fea)
        edge_index, edge_attr = self._build_graph(atom_fea, nbr_fea, nbr_fea_idx)
        edge_feat = (
            self.edge_proj(edge_attr)
            if edge_attr.numel() > 0
            else torch.zeros(0, self.edge_proj.out_features, device=atom_fea.device)
        )
        for layer in self.layers:
            node_feat, edge_feat = layer(node_feat, edge_feat, edge_index)
        crys_feat = self.pooling(node_feat, crystal_atom_idx)
        crys_feat = self.readout(crys_feat)
        crys_feat = self.dropout(crys_feat)
        out = self.fc_out(crys_feat)
        if self.classification:
            out = self.logsoftmax(out)
        return out
