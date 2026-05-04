from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .env import HBondGraphEnv, SRGConfig
from .ncg import ncg_reward_correction, spectral_metrics
from .policy import ActionPolicy, action_features
from .braid import BraidTracker, braid_reward_correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--n_solvent', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--gamma', type=float, default=0.96)
    ap.add_argument('--entropy_coef', type=float, default=0.01)
    ap.add_argument('--lambda_ncg', type=float, default=0.05)
    ap.add_argument('--lambda_gap', type=float, default=0.01)
    ap.add_argument('--lambda_braid', type=float, default=0.01)
    ap.add_argument('--lambda_writhe', type=float, default=0.002)
    ap.add_argument('--lambda_braid_entropy', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='runs/rl_braid_demo')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    policy = ActionPolicy(); opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    hist = []; best_energy = 1e18

    for ep in range(args.episodes):
        env = HBondGraphEnv(SRGConfig(n_solvent=args.n_solvent, seed=args.seed + ep))
        braid = BraidTracker(env.n_nodes)
        logps, rewards, entropies = [], [], []
        raw_rewards = []
        for t in range(args.horizon):
            acts = env.possible_actions(limit_pairs=240)
            x = action_features(env, acts)
            logits = policy(x)
            dist = Categorical(logits=logits)
            idx = dist.sample(); action = acts[int(idx)]
            state, reward, _ = env.step(action)
            braid.add_action(state['positions'], action, step=t)
            corr_ncg, _ = ncg_reward_correction(state, args.lambda_ncg, args.lambda_gap)
            corr_braid, _ = braid_reward_correction(braid.metrics(), args.lambda_braid, args.lambda_writhe, args.lambda_braid_entropy)
            shaped = reward + corr_ncg + corr_braid
            logps.append(dist.log_prob(idx)); entropies.append(dist.entropy())
            rewards.append(float(shaped)); raw_rewards.append(float(reward))
        G = 0.0; returns = []
        for r in reversed(rewards):
            G = r + args.gamma * G; returns.append(G)
        returns.reverse()
        R = torch.tensor(returns, dtype=torch.float32)
        if len(R) > 1:
            R = (R - R.mean()) / (R.std() + 1e-6)
        loss = -torch.stack(logps).mul(R).sum() - args.entropy_coef * torch.stack(entropies).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        st = env.state(); met = spectral_metrics(st); bm = braid.metrics()
        best_energy = min(best_energy, st['energy'])
        row = {
            'episode': ep,
            'loss': float(loss.detach()),
            'return': float(sum(rewards)),
            'raw_return': float(sum(raw_rewards)),
            'energy': float(st['energy']),
            'best_energy': float(best_energy),
            'ncg_smoothness': met.ncg_smoothness,
            'dirac_gap': met.spectral_gap,
            'dirac_radius': met.dirac_radius,
            'laplacian_gap': met.laplacian_gap,
            'n_edges': met.n_edges,
            'solvation_shell_degree': int(st['solvation_shell_degree']),
            'braid_raw_length': bm.raw_length,
            'braid_reduced_length': bm.reduced_length,
            'braid_writhe': bm.writhe,
            'braid_entropy': bm.generator_entropy,
            'braid_permutation_disorder': bm.permutation_disorder,
            'braid_word': braid.word_string(),
        }
        hist.append(row)
        if ep % max(1, args.episodes // 10) == 0:
            print(f"[{ep}] R={row['return']:.3f} E={row['energy']:.3f} braid={row['braid_reduced_length']} smooth={row['ncg_smoothness']:.3f}")
    torch.save(policy.state_dict(), out / 'policy.pt')
    with open(out / 'history.json', 'w') as f: json.dump(hist, f, indent=2)
    print('wrote', out)

if __name__ == '__main__':
    main()
