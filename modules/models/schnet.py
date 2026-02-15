import torch
from torch import nn
from typing import List


class _InteractionBlock(nn.Module):
    """Continuous-filter interaction block used inside SchNet."""

    def __init__(self, hidden_dim: int, filter_dim: int, nbr_fea_len: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(nbr_fea_len, filter_dim),
            nn.SiLU(),
            nn.Linear(filter_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_repr: torch.Tensor, nbr_fea: torch.Tensor, nbr_fea_idx: torch.Tensor) -> torch.Tensor:
        # node_repr: [N, hidden], nbr_fea: [N, M, Fn], nbr_fea_idx: [N, M]
        neighbor_repr = node_repr[nbr_fea_idx]  # [N, M, hidden]
        filters = self.filter_net(nbr_fea)      # [N, M, hidden]
        message = (neighbor_repr * filters).sum(dim=1)
        return self.update(message)


class CrystalGraphSchNet(nn.Module):
    """SchNet-style model that consumes CGCNN graph tensors."""

    def __init__(
        self,
        atom_fea_len: int,
        nbr_fea_len: int,
        hidden_dim: int = 128,
        filter_dim: int = 64,
        num_interactions: int = 3,
        readout: str = "mean",
        task: str = "regression",
        num_classes: int = 2,
    ):
        super().__init__()
        if readout not in {"mean", "sum", "max"}:
            raise ValueError(f"Unsupported readout '{readout}'")
        if task not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task '{task}'")

        self.readout = readout
        self.task = task

        self.atom_embed = nn.Sequential(
            nn.Linear(atom_fea_len, hidden_dim),
            nn.SiLU(),
        )
        self.interactions = nn.ModuleList([
            _InteractionBlock(hidden_dim, filter_dim, nbr_fea_len)
            for _ in range(num_interactions)
        ])
        self.activation = nn.SiLU()

        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _pool_crystal(self, node_repr: torch.Tensor, crystal_atom_idx: List[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for idx in crystal_atom_idx:
            idx = idx.long()
            crystal_nodes = torch.index_select(node_repr, 0, idx)
            if self.readout == "mean":
                pooled.append(crystal_nodes.mean(dim=0))
            elif self.readout == "sum":
                pooled.append(crystal_nodes.sum(dim=0))
            else:
                pooled.append(crystal_nodes.max(dim=0).values)
        return torch.stack(pooled, dim=0)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        node_repr = self.atom_embed(atom_fea)
        for block in self.interactions:
            node_repr = node_repr + block(node_repr, nbr_fea, nbr_fea_idx)
            node_repr = self.activation(node_repr)

        crystal_repr = self._pool_crystal(node_repr, crystal_atom_idx)
        output = self.head(crystal_repr)
        if self.task == "regression":
            return output.view(-1)
        return output
