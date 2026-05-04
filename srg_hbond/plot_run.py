from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run_dir', required=True); args=ap.parse_args()
    rd=Path(args.run_dir)
    hist=json.load(open(rd/'history.json'))
    x=[r.get('step', r.get('episode')) for r in hist]
    for key in ['best_energy','energy','mean_energy','ncg_smoothness','laplacian_gap','dirac_radius','solvation_shell_degree']:
        if key in hist[0]:
            y=[r[key] for r in hist]
            plt.figure(figsize=(8,4)); plt.plot(x,y); plt.xlabel('step/episode'); plt.ylabel(key); plt.tight_layout(); plt.savefig(rd/f'{key}.png', dpi=150); plt.close()
    print('wrote plots to', rd)
if __name__=='__main__': main()
