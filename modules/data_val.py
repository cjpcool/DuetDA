import torch
import torch.nn as nn

class TinyTransformerBackbone(nn.Module):
    """
    A tiny Transformer encoder backbone for low-parameter settings.

    Supports:
      - x: (B, input_dim)  -> optionally chunk into tokens
      - x: (B, L, token_dim) -> directly as tokens
    """
    def __init__(
        self,
        input_dim=256,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_ff=128,
        dropout=0.1,
        max_len=64,
        chunk_size=16,
        query_dim=256,
        use_pos_emb=False,
        cross_attn=False,
        pooling="cls",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.chunk_size = chunk_size
        self.use_pos_emb = use_pos_emb
        self.cross_attn = cross_attn
        self.pooling = pooling

        
        if self.cross_attn:
            self.q_proj = nn.Linear(query_dim, d_model)
            self.q_norm = nn.LayerNorm(d_model)
        
        self.token_dim = chunk_size if chunk_size is not None else input_dim

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")

        self.input_proj = nn.Linear(self.token_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Use MultiheadAttention instead of TransformerEncoderLayer
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.ReLU(),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers * 2)])
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, query=None) -> torch.Tensor:
        """
        Returns pooled representation: (B, d_model)
        """
        B = max(x.size(0), query.size(0) if query is not None else x.size(0))
        if x.dim() == 2:
            if self.chunk_size is not None:
                if x.size(-1) % self.chunk_size != 0:
                    raise ValueError(
                        f"input_dim ({x.size(-1)}) must be divisible by chunk_size ({self.chunk_size})."
                    )
                if x.size(0) != B:
                    x = x.repeat(B // x.size(0), 1)
                L = x.size(-1) // self.chunk_size
                x = x.view(B, L, self.chunk_size)
            else:
                x = x.unsqueeze(1)

        if x.dim() != 3:
            raise ValueError(f"Expected x dim 2 or 3, got {x.dim()}.")

        B, L, _ = x.shape
        x = self.input_proj(x)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.use_pos_emb:
            if x.size(1) > self.pos_emb.size(1):
                raise ValueError(
                    f"Sequence length {x.size(1)} exceeds max_len+1 {self.pos_emb.size(1)}. "
                    f"Increase max_len."
                )
            x = x + self.pos_emb[:, : x.size(1), :]

        if query is not None:
            if query.dim() == 2: # (B, query_dim)
                query = query.unsqueeze(1)  # (B, 1, query_dim)
            if query.size(0) !=  B:
                query = query.repeat(B, 1, 1)
        else:
            query = x
            
        if self.cross_attn:
            q_emb = self.q_proj(query)

        else:
            q_emb = x

        for attn, ff, norm1, norm2 in zip(
            self.attention_layers,
            self.feedforward_layers,
            self.norms[::2],
            self.norms[1::2]
        ):
            x_norm = norm1(x)

            if self.cross_attn:
                q_norm = self.q_norm(q_emb)
                attn_out, _ = attn(q_norm, x_norm, x_norm)  # (B,1,d)
                x_cls = x[:, :1, :] + attn_out
                x = torch.cat([x_cls, x[:, 1:, :]], dim=1)
            else:
                attn_out, _ = attn(x_norm, x_norm, x_norm)
                x = x + attn_out

            x_norm = norm2(x)

            if self.cross_attn:
                ff_cls = ff(x_norm[:, :1, :])
                x_cls = x[:, :1, :] + ff_cls
                x = torch.cat([x_cls, x[:, 1:, :]], dim=1)
            else:
                x = x + ff(x_norm)
                x = self.norm(x)

        if self.pooling == "cls":
            return x[:, 0, :]
        elif self.pooling == "mean":
            return x[:, 1:, :].mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


class MLPBackbone(nn.Module):
    """Simple MLP backbone."""
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1): 
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
class GateNet(nn.Module):
    def __init__(
        self, status_dim=6, feat_dim=6, hidden_dim=32, chunk_size=16, max_len=64, backbone='transformer', temperature = 5.0, clamp_min=0.0, clamp_max=1.0
    ):
        super().__init__()        
        if backbone == 'transformer':
            self.backbone = TinyTransformerBackbone(
                input_dim=status_dim,
                query_dim=feat_dim,
                cross_attn=True,
                d_model=hidden_dim,   # e.g., 64
                nhead=1,
                num_layers=1,
                dim_ff=hidden_dim * 2,  # small FFN
                dropout=0.1,
                max_len=max_len,
                chunk_size=chunk_size,        # (B,256)->(B,16,16)
                use_pos_emb=True,
                pooling="cls"
            )
        elif backbone == 'mlp':
            self.backbone= MLPBackbone(input_dim=2*status_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,num_layers=4)
            self.feat_proj = nn.Linear(feat_dim, status_dim)
        self.status_dim = status_dim
        self.feat_dim = feat_dim
        self.output = nn.Linear(hidden_dim, 2)
        self.temperature = temperature
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        
    def forward(self, status: torch.Tensor, x: torch.Tensor):
        """
        status: (B, status_dim)
        x:      (B, input_dim) or (B, L, token_dim)

        Returns:
            alpha_phy: (B, 1)
            alpha_gen: (B, 1)
            aux: dict
        """
        if status.dim() == 1:
            status = status.unsqueeze(0)
        
        
        if isinstance(self.backbone, TinyTransformerBackbone):
            h = self.backbone(status, query=x)
        elif isinstance(self.backbone, MLPBackbone):
            if x.dim() == 3:
                # x: (B, L, feat_dim) -> mean pool over L
                x_pooled = x.mean(dim=1)
            elif x.dim() == 2:
                # x: (B, feat_dim) -> use as-is
                x_pooled = x
            else:
                raise ValueError(f"Unsupported input shape for x in GateNet with MLPBackbone: {x.shape}")
            x_proj = self.feat_proj(x_pooled)  # (B, status_dim)
            h = self.backbone(torch.cat([status, x_proj], dim=-1))  # (B, hidden_dim)

        logits = self.output(h)  # (B, 2)


        weights = torch.softmax(logits / self.temperature, dim=-1)
        alpha_phy = weights[:, 0].view(-1, 1)
        alpha_gen = weights[:, 1].view(-1, 1)
        
        # weights = torch.sigmoid(logits / self.temperature)
        # alpha_gen = weights[:, 1].view(-1, 1)
        # alpha_phy = weights[:, 0].view(-1, 1)
        # alpha_gen = alpha_gen / (alpha_gen + alpha_phy + 1e-8)
        # alpha_phy = alpha_phy / (alpha_gen + alpha_phy + 1e-8)

        # if (self.clamp_min > 0.0) or (self.clamp_max < 1.0):
        #     alpha_phy = alpha_phy.clamp(self.clamp_min, self.clamp_max)
        #     alpha_gen = alpha_gen.clamp(self.clamp_min, self.clamp_max)


        return alpha_phy, alpha_gen

class DataValPhysics(nn.Module):
    """Transformer-based physics valuation head."""
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=1,chunk_size=16, max_len=64, backbone='mlp'):
        super().__init__()
        # hidden_dim here maps to d_model to keep signature similar & params small
        if backbone == 'transformer':
            self.backbone = TinyTransformerBackbone(
                input_dim=input_dim,
                d_model=hidden_dim,   # e.g., 64
                nhead=1,
                num_layers=1,
                dim_ff=hidden_dim * 2,  # small FFN
                dropout=0.1,
                max_len=max_len,
                chunk_size=chunk_size,        # (B,256)->(B,16,16)
                use_pos_emb=True,
                pooling="cls"
            )
        elif backbone == 'mlp':
            self.backbone= MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,num_layers=4)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.out_func = nn.Sigmoid()

    def forward(self, x):
        h = self.backbone(x)
        y = self.head(h)
        return self.out_func(y)


class DataValGen(nn.Module):
    """Transformer-based generative valuation head."""
    def __init__(self, input_dim=256, hidden_dim=64, output_dim=1, chunk_size=16, max_len=64, backbone='mlp'):
        super().__init__()
        if backbone == 'transformer':
            self.backbone = TinyTransformerBackbone(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=1,
                num_layers=1,
                dim_ff=hidden_dim * 2,
                dropout=0.1,
                max_len=max_len,
                chunk_size=chunk_size,
                use_pos_emb=True,
                pooling="cls"
            )
        elif backbone == 'mlp':
            self.backbone= MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,num_layers=4)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.out_func = nn.Sigmoid()

    def forward(self, x):
        h = self.backbone(x)
        y = self.head(h)
        return self.out_func(y)


class DualDataValuator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=32, output_dim=1, task='regression', num_classes=None, chunk_size=10, max_len=600, ema_stat_len=6,gate_clamp_min=0.05, gate_clamp_max=0.95):
        super().__init__()
        assert input_dim % chunk_size == 0, f"input_dim{input_dim} must be divisible by chunk_size{chunk_size}."
        assert input_dim // chunk_size <= max_len, "Number of tokens exceeds max_len."
        self.physics_head = DataValPhysics(input_dim, hidden_dim, output_dim, chunk_size, max_len, backbone='transformer')
        self.gen_head = DataValGen(input_dim, hidden_dim, output_dim, chunk_size, max_len, backbone='transformer')
        self.gate = GateNet(status_dim=ema_stat_len, feat_dim=input_dim, hidden_dim=hidden_dim, clamp_min=gate_clamp_min, clamp_max=gate_clamp_max, chunk_size=3, max_len=max_len, backbone='transformer')
    def forward(self, x, weight_phy=None, weight_gen=None,status=None, ret_sep=False):
        val_phy = self.physics_head(x)
        val_gen = self.gen_head(x)

        if status is None:
            alpha_phy, alpha_gen = 0.5, 0.5
        else:
            alpha_phy, alpha_gen = self.gate(status, x) 

        weight_phy = alpha_phy if weight_phy is None else weight_phy
        weight_gen = alpha_gen if weight_gen is None else weight_gen

        if ret_sep:
            return val_phy, val_gen, weight_phy, weight_gen
        return weight_phy * val_phy + weight_gen * val_gen
