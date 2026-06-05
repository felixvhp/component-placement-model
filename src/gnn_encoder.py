

import torch
from torch import nn


class GNNConstraintEncoder(nn.Module):
    def __init__(
        self,
        max_n=25,
        edge_dim=3,
        node_dim=64,
        hidden_dim=128,
        d_model=256,
        n_layers=3,
    ):
        super().__init__()
        self.max_n = max_n
        self.node_dim = node_dim

        # Learned per-slot node embeddings (one per box position)
        self.node_embed = nn.Embedding(max_n, node_dim)

        # takes (sender_state, receiver_state, edge_feat)
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, node_dim),
            )
            for _ in range(n_layers)
        ])

        # Update network: combines old node state with aggregated message.
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * node_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, node_dim),
            )
            for _ in range(n_layers)
        ])

        # Final projection from pooled node states to the conditioning vector.
        self.out_proj = nn.Sequential(
            nn.Linear(node_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # zero-init the final projection so the encoder starts as no-op
        nn.init.zeros_(self.out_proj[-1].weight)
        nn.init.zeros_(self.out_proj[-1].bias)

    def forward(self, target):
        """
        target: [batch, N, N, edge_dim] — signed-relation constraint tensor.
                N ≤ max_n; the function handles variable N via a node mask.

        Returns: [batch, d_model] — conditioning vector.
        """
        batch_size, n, _, _ = target.shape
        device = target.device

        # node initialization: use learned embeddings for slots 0..n-1
        slot_ids = torch.arange(n, device=device)
        h = self.node_embed(slot_ids)
        h = h.unsqueeze(0).expand(batch_size, -1, -1)

        # Edge features are just the target tensor itself [batch, n, n, edge_dim]
        edge_feat = target

        # Node mask: all nodes valid since we're passing exactly N (not padded).
        # If you ever pad: mask nonexistent nodes here.

        for msg_mlp, upd_mlp in zip(self.message_mlps, self.update_mlps):
            h_send = h.unsqueeze(1).expand(-1, n, -1, -1)
            h_recv = h.unsqueeze(2).expand(-1, -1, n, -1)

            msg_input = torch.cat([h_send, h_recv, edge_feat], dim=-1)
            messages = msg_mlp(msg_input)

            eye = torch.eye(n, device=device).unsqueeze(0).unsqueeze(-1)
            messages = messages * (1 - eye)
            aggregated = messages.sum(dim=2) / max(n - 1, 1)

            # update node state
            h = upd_mlp(torch.cat([h, aggregated], dim=-1))

        # pool node states into a single conditioning vector
        pooled = h.mean(dim=1)
        return self.out_proj(pooled)
