from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any, Dict, List

from .metrics_bundle import load_metrics_jsonl


def _import_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def load_history(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.is_dir():
        mj = p / "metrics_bundle.jsonl"
        if mj.exists():
            return load_metrics_jsonl(mj)
        for name in ("frames.json", "history.json", "frames_live.json"):
            q = p / name
            if q.exists():
                return json.loads(q.read_text())
        raise FileNotFoundError(f"No metrics_bundle.jsonl / frames.json / history.json in {p}")
    return json.loads(p.read_text())


def generate_png_report(run_dir: str | Path, history: List[Dict[str, Any]] | None = None) -> List[str]:
    run_dir = Path(run_dir)
    fig_dir = run_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    if history is None:
        history = load_history(run_dir)
    if not history:
        return []
    plt = _import_mpl()
    written: List[str] = []

    def seq(key, default=0.0):
        return [float(r.get(key, default) or 0.0) for r in history]

    steps = [int(r.get('step', i)) for i, r in enumerate(history)]
    plots = [
        ('energy.png', 'Free energy', seq('energy'), 'energy'),
        ('reward.png', 'Reward', seq('reward'), 'reward'),
        ('ncg_smoothness.png', 'NCG smoothness', seq('ncg_smoothness'), 'smoothness'),
        ('spectral_gaps.png', 'Spectral gaps', None, 'gap'),
        ('braid_length.png', 'Braid reduced length', seq('braid_reduced_length'), 'length'),
        ('edges.png', 'Number of graph edges', seq('n_edges'), 'edges'),
    ]
    for fname, title, y, ylabel in plots:
        plt.figure(figsize=(8, 4.5))
        if fname == 'spectral_gaps.png':
            plt.plot(steps, seq('dirac_gap'), label='Dirac gap')
            plt.plot(steps, seq('laplacian_gap'), label='Laplacian gap')
            plt.legend()
        else:
            plt.plot(steps, y)
        plt.title(title); plt.xlabel('step'); plt.ylabel(ylabel); plt.tight_layout()
        out = fig_dir / fname; plt.savefig(out, dpi=160); plt.close(); written.append(str(out))

    # energy terms stacked-ish individual lines
    term_keys = sorted({k for r in history for k in (r.get('energy_terms') or {}).keys()})
    if term_keys:
        plt.figure(figsize=(9, 5))
        for k in term_keys:
            vals = [float((r.get('energy_terms') or {}).get(k, 0.0)) for r in history]
            plt.plot(steps, vals, label=k)
        plt.title('Energy terms breakdown')
        plt.xlabel('step'); plt.ylabel('term contribution')
        plt.legend(fontsize=8, ncol=2); plt.tight_layout()
        out = fig_dir / 'energy_terms.png'; plt.savefig(out, dpi=170); plt.close(); written.append(str(out))

    # RGB absorbance
    if any('absorbance_rgb' in r for r in history):
        vals = [r.get('absorbance_rgb', [0, 0, 0]) for r in history]
        plt.figure(figsize=(8, 4.5))
        for idx, name in enumerate(['R', 'G', 'B']):
            plt.plot(steps, [float(v[idx]) if len(v) > idx else 0.0 for v in vals], label=name)
        plt.title('Colorimeter absorbance proxy')
        plt.xlabel('step'); plt.ylabel('absorbance proxy'); plt.legend(); plt.tight_layout()
        out = fig_dir / 'absorbance_rgb.png'; plt.savefig(out, dpi=160); plt.close(); written.append(str(out))

    if any(isinstance(r.get("absorbance_spectrum"), dict) for r in history):
        keys = sorted({k for r in history for k in (r.get("absorbance_spectrum") or {}).keys()})
        if keys:
            plt.figure(figsize=(8, 4.5))
            for nm in keys:
                ys = [float((r.get("absorbance_spectrum") or {}).get(nm, 0.0)) for r in history]
                plt.plot(steps, ys, label=f"{nm} nm")
            plt.title("Multi-band absorbance proxy")
            plt.xlabel("step")
            plt.ylabel("mean absorption proxy")
            plt.legend(fontsize=8)
            plt.tight_layout()
            out = fig_dir / "absorbance_spectrum.png"
            plt.savefig(out, dpi=160)
            plt.close()
            written.append(str(out))

    # electrolyte metrics
    if any(r.get('electrolyte') for r in history):
        def eseq(key):
            vals=[]
            for r in history:
                e=r.get('electrolyte') or {}
                vals.append(float(e.get(key, 0.0) or 0.0))
            return vals
        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, eseq('ionic_strength_M'), label='ionic strength M')
        plt.plot(steps, eseq('gamma_mean'), label='gamma mean')
        plt.title('Electrolyte activity metrics')
        plt.xlabel('step'); plt.legend(); plt.tight_layout()
        out = fig_dir / 'electrolyte_activity.png'; plt.savefig(out, dpi=160); plt.close(); written.append(str(out))

        plt.figure(figsize=(8, 4.5))
        plt.plot(steps, eseq('debye_length_nm'), label='Debye length nm')
        plt.title('Debye screening length')
        plt.xlabel('step'); plt.ylabel('nm'); plt.legend(); plt.tight_layout()
        out = fig_dir / 'debye_length.png'; plt.savefig(out, dpi=160); plt.close(); written.append(str(out))

        if any('conductivity_s_cm_proxy' in (r.get('electrolyte') or {}) for r in history):
            plt.figure(figsize=(8, 4.5))
            plt.plot(steps, eseq('molar_conductivity_s_cm2_mol'), label='molar conductivity proxy')
            plt.plot(steps, eseq('conductivity_s_cm_proxy'), label='specific conductivity proxy')
            plt.title('Onsager/Kohlrausch conductivity proxy')
            plt.xlabel('step'); plt.legend(); plt.tight_layout()
            out = fig_dir / 'conductivity_proxy.png'; plt.savefig(out, dpi=160); plt.close(); written.append(str(out))

    # final graph snapshot if positions exist
    last = history[-1]
    if 'positions' in last and 'edges' in last and 'types' in last:
        palette = {'water':'#4c8dff','Na+':'#ffcc33','Cl-':'#88dd88','polar':'#b36bff','hydrophobic':'#444444','dye_A':'#ff4fc3','dye_B':'#00bcd4','solute':'#e74c3c'}
        pos = last['positions']; types = last['types']
        plt.figure(figsize=(7, 7))
        for e in last.get('edges', []):
            i, j = int(e[0]), int(e[1])
            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], linewidth=0.7, alpha=0.35)
        for typ in sorted(set(types)):
            xs = [pos[i][0] for i,t in enumerate(types) if t == typ]
            ys = [pos[i][1] for i,t in enumerate(types) if t == typ]
            plt.scatter(xs, ys, s=28 if typ == 'water' else 70, label=typ, c=palette.get(typ, '#777777'))
        plt.title(f'Final graph snapshot: step {last.get("step", len(history)-1)}')
        plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05); plt.legend(fontsize=8, loc='best'); plt.tight_layout()
        out = fig_dir / 'final_graph.png'; plt.savefig(out, dpi=180); plt.close(); written.append(str(out))

    (fig_dir / 'manifest.json').write_text(json.dumps({'figures': written}, indent=2))
    return written
