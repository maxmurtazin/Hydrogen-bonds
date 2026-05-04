from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple, Any

import numpy as np

Node = int
Edge = Tuple[int, int]
Action = Tuple[str, int, int]


@dataclass
class SRGConfig:
    n_solvent: int = 24
    box: float = 1.0
    max_hbond_dist: float = 0.34
    solute_radius: float = 0.26
    hbond_energy: float = -1.0
    solvation_energy: float = -1.5
    steric_penalty: float = 0.15
    entropy_weight: float = 0.08
    max_degree: int = 4
    seed: int = 7


class HBondGraphEnv:
    """Toy SRG environment: dynamic H-bond graph around one solute node.

    Node 0 is the solute. Nodes 1..n are solvent molecules. Positions are fixed in
    this MVP; actions form/break edges. Energy favors plausible H-bond and solvation
    contacts while penalizing over-coordination.
    """

    def __init__(self, cfg: SRGConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.n_nodes = cfg.n_solvent + 1
        self.reset()

    def reset(self) -> Dict[str, Any]:
        c = self.cfg
        self.pos = np.zeros((self.n_nodes, 2), dtype=np.float32)
        self.pos[0] = np.array([0.5, 0.5], dtype=np.float32)
        for i in range(1, self.n_nodes):
            self.pos[i] = np.array([self.rng.random(), self.rng.random()], dtype=np.float32)
        self.edges: Dict[Edge, float] = {}
        # initialize likely contacts
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                d = self.distance(i, j)
                if i == 0 or j == 0:
                    p = 0.45 if d < c.solute_radius * 1.35 else 0.02
                else:
                    p = 0.35 if d < c.max_hbond_dist else 0.03
                if self.rng.random() < p:
                    self.edges[(i, j)] = self.edge_weight(i, j)
        return self.state()

    def clone(self) -> 'HBondGraphEnv':
        other = HBondGraphEnv(self.cfg)
        other.pos = self.pos.copy()
        other.edges = dict(self.edges)
        return other

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.pos[i] - self.pos[j]) + 1e-8)

    def edge_weight(self, i: int, j: int) -> float:
        d = self.distance(i, j)
        if i == 0 or j == 0:
            sigma = self.cfg.solute_radius
        else:
            sigma = self.cfg.max_hbond_dist
        # smooth contact weight: near contacts stronger
        return float(math.exp(-(d / sigma) ** 2))

    def canonical_edge(self, i: int, j: int) -> Edge:
        return (i, j) if i < j else (j, i)

    def degree(self, i: int) -> int:
        return sum(1 for e in self.edges if i in e)

    def possible_actions(self, limit_pairs: int | None = None) -> List[Action]:
        actions: List[Action] = []
        pairs = [(i, j) for i in range(self.n_nodes) for j in range(i + 1, self.n_nodes)]
        if limit_pairs and len(pairs) > limit_pairs:
            pairs = self.rng.sample(pairs, limit_pairs)
        for i, j in pairs:
            e = (i, j)
            if e in self.edges:
                actions.append(('break', i, j))
            else:
                # only allow plausible local contacts to avoid all-to-all nonsense
                d = self.distance(i, j)
                if i == 0 or j == 0:
                    if d < self.cfg.solute_radius * 1.8:
                        actions.append(('form', i, j))
                elif d < self.cfg.max_hbond_dist * 1.6:
                    actions.append(('form', i, j))
        if not actions:
            actions.append(('noop', 0, 0))
        return actions

    def step(self, action: Action) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        e0 = self.free_energy()
        kind, i, j = action
        if kind == 'form':
            self.edges[self.canonical_edge(i, j)] = self.edge_weight(i, j)
        elif kind == 'break':
            self.edges.pop(self.canonical_edge(i, j), None)
        e1 = self.free_energy()
        reward = -(e1 - e0)
        info = {'energy': e1, 'delta_energy': e1 - e0, 'reward': reward}
        return self.state(), reward, info

    def free_energy(self) -> float:
        c = self.cfg
        e_bond = 0.0
        e_solv = 0.0
        steric = 0.0
        for (i, j), w in self.edges.items():
            if i == 0 or j == 0:
                e_solv += c.solvation_energy * w
            else:
                e_bond += c.hbond_energy * w
        for i in range(self.n_nodes):
            over = max(0, self.degree(i) - c.max_degree)
            steric += c.steric_penalty * over * over
        # entropy proxy: too few or too many edges reduce configurational freedom
        m = len(self.edges)
        entropy_bonus = -c.entropy_weight * math.log(1.0 + m)
        return float(e_bond + e_solv + steric + entropy_bonus)

    def node_features(self) -> np.ndarray:
        feats = np.zeros((self.n_nodes, 5), dtype=np.float32)
        center = self.pos[0]
        for i in range(self.n_nodes):
            feats[i, 0] = 1.0 if i == 0 else 0.0
            feats[i, 1] = self.degree(i) / max(1, self.cfg.max_degree)
            feats[i, 2] = float(np.linalg.norm(self.pos[i] - center))
            feats[i, 3] = self.pos[i, 0]
            feats[i, 4] = self.pos[i, 1]
        return feats

    def adjacency(self) -> np.ndarray:
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for (i, j), w in self.edges.items():
            adj[i, j] = adj[j, i] = w
        return adj

    def solvation_shell_degree(self) -> int:
        return self.degree(0)

    def state(self) -> Dict[str, Any]:
        return {
            'n_nodes': self.n_nodes,
            'positions': self.pos.copy(),
            'edges': dict(self.edges),
            'adjacency': self.adjacency(),
            'features': self.node_features(),
            'energy': self.free_energy(),
            'solvation_shell_degree': self.solvation_shell_degree(),
        }
