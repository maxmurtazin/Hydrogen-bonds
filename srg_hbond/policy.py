from __future__ import annotations
import torch
from torch import nn


class ActionPolicy(nn.Module):
    """Small MLP scoring candidate actions from simple pair features."""
    def __init__(self, in_dim: int = 13, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def action_features(env, actions):
    feats = env.node_features()
    rows = []
    for kind, i, j in actions:
        kind_code = 0.0 if kind == 'break' else (1.0 if kind == 'form' else -1.0)
        d = env.distance(i, j) if i != j else 0.0
        exists = 1.0 if env.canonical_edge(i, j) in env.edges else 0.0
        row = [kind_code, d, exists]
        row += feats[i].tolist()
        row += feats[j].tolist()
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)
