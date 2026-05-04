from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Dict

import numpy as np

from .env import HBondGraphEnv, SRGConfig, Action
from .ncg import ncg_reward_correction, spectral_metrics
from .braid import BraidTracker, braid_reward_correction


def action_key(a: Action) -> str:
    return f'{a[0]}:{a[1]}:{a[2]}'


def choose_action(env: HBondGraphEnv, pher: Dict[str, float], alpha: float, beta: float, temp: float):
    acts = env.possible_actions(limit_pairs=240)
    weights = []
    e0 = env.free_energy()
    for a in acts:
        clone = env.clone(); clone.step(a)
        de = clone.free_energy() - e0
        eta = math.exp(-de / max(temp, 1e-6))
        tau = pher.get(action_key(a), 1.0)
        weights.append((tau ** alpha) * (eta ** beta))
    s = sum(weights)
    if s <= 0 or not math.isfinite(s):
        return random.choice(acts)
    r = random.random() * s; acc = 0.0
    for a, w in zip(acts, weights):
        acc += w
        if acc >= r:
            return a
    return acts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--ants', type=int, default=32)
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--n_solvent', type=int, default=24)
    ap.add_argument('--rho', type=float, default=0.08)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=2.0)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--lambda_ncg', type=float, default=0.05)
    ap.add_argument('--lambda_gap', type=float, default=0.01)
    ap.add_argument('--lambda_braid', type=float, default=0.01)
    ap.add_argument('--lambda_writhe', type=float, default=0.002)
    ap.add_argument('--lambda_braid_entropy', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='runs/aco_braid_demo')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = SRGConfig(n_solvent=args.n_solvent, seed=args.seed)
    base_env = HBondGraphEnv(cfg)
    pher: Dict[str, float] = {}
    history = []
    best = {'score': -1e18, 'energy': 1e18, 'edges': None, 'state': None, 'braid_word': '1'}

    for step in range(args.steps):
        deposits: Dict[str, float] = {}
        scores, energies = [], []
        best_step = None
        for ant in range(args.ants):
            env = base_env.clone()
            braid = BraidTracker(env.n_nodes)
            total = 0.0; used = []
            for h in range(args.horizon):
                a = choose_action(env, pher, args.alpha, args.beta, args.temperature)
                state, reward, _ = env.step(a)
                braid.add_action(state['positions'], a, step=h)
                corr_ncg, _ = ncg_reward_correction(state, args.lambda_ncg, args.lambda_gap)
                corr_braid, bm = braid_reward_correction(braid.metrics(), args.lambda_braid, args.lambda_writhe, args.lambda_braid_entropy)
                total += reward + corr_ncg + corr_braid
                used.append(action_key(a))
            energy = env.free_energy()
            scores.append(total); energies.append(energy)
            if total > best['score']:
                best = {'score': total, 'energy': energy, 'edges': list(env.edges.items()), 'state': env.state(), 'braid_word': braid.word_string(), 'braid_events': braid.events}
            if best_step is None or total > best_step[0]:
                best_step = (total, used, energy, env.state(), braid.copy())
            for k in used:
                deposits[k] = deposits.get(k, 0.0) + max(0.0, total)
        for k in list(pher.keys()):
            pher[k] *= (1.0 - args.rho)
            if pher[k] < 1e-5: del pher[k]
        for k, v in deposits.items():
            pher[k] = pher.get(k, 1.0) + v / max(1, args.ants)
        st = best_step[3]; braid = best_step[4]
        met = spectral_metrics(st); bm = braid.metrics()
        row = {
            'step': step,
            'mean_score': float(np.mean(scores)),
            'best_step_score': float(best_step[0]),
            'mean_energy': float(np.mean(energies)),
            'best_energy': float(best['energy']),
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
            'pheromone_size': len(pher),
        }
        history.append(row)
        if step % max(1, args.steps // 10) == 0:
            print(f"[{step}] score={row['best_step_score']:.3f} E={row['best_energy']:.3f} braid={row['braid_reduced_length']} smooth={row['ncg_smoothness']:.3f}")

    with open(out / 'history.json', 'w') as f: json.dump(history, f, indent=2)
    with open(out / 'best.json', 'w') as f:
        json.dump({'score': best['score'], 'energy': best['energy'], 'braid_word': best['braid_word'], 'braid_events': best['braid_events'], 'edges': [list(e)+[w] for (e,w) in best['edges']]}, f, indent=2)
    print('wrote', out)

if __name__ == '__main__':
    main()
