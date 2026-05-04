"""
Publication-style figure generation for SRG Chemical Computing Platform v6.1.

Loads experiment history from ``metrics_bundle.jsonl`` (preferred), ``frames.json``,
``history.json``, or ``frames_live.json``, then writes PNG (and optionally PDF) plots.

CLI::

    python3 -m srg_hbond.figures_pipeline --run_dir runs/exp_name
    python3 -m srg_hbond.figures_pipeline --compare runs/a runs/b
    python3 -m srg_hbond.figures_pipeline --all runs/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Data loading (single pass per file where possible)
# -----------------------------------------------------------------------------


def _resolve_history_path(run_dir: Path) -> Tuple[str, Path]:
    """Return (kind, path) for the first available history source."""
    run_dir = run_dir.resolve()
    candidates = [
        ("jsonl", run_dir / "metrics_bundle.jsonl"),
        ("json", run_dir / "frames.json"),
        ("json", run_dir / "history.json"),
        ("json", run_dir / "frames_live.json"),
    ]
    for kind, p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return kind, p
    raise FileNotFoundError(
        f"No metrics_bundle.jsonl, frames.json, history.json, or frames_live.json in {run_dir}"
    )


def load_history_efficient(run_dir: str | Path) -> List[Dict[str, Any]]:
    """Load full history once. JSONL is streamed line-by-line to avoid peak memory during parse."""
    run_dir = Path(run_dir)
    kind, path = _resolve_history_path(run_dir)
    if kind == "jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out
    return json.loads(path.read_text(encoding="utf-8"))


def iter_history_records(run_dir: str | Path) -> Iterator[Dict[str, Any]]:
    """Stream records without building a full list (for future chunked stats)."""
    run_dir = Path(run_dir)
    kind, path = _resolve_history_path(run_dir)
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        for rec in json.loads(path.read_text(encoding="utf-8")):
            yield rec


def discover_run_directories(root: str | Path) -> List[Path]:
    """Subdirectories of ``root`` that look like experiment outputs."""
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    found: List[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            _resolve_history_path(d)
        except FileNotFoundError:
            continue
        found.append(d)
    return found


# -----------------------------------------------------------------------------
# Series extraction
# -----------------------------------------------------------------------------


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


def extract_series(history: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build aligned arrays for plotting from frame records."""
    n = len(history)
    steps: List[int] = []
    energy: List[float] = []
    ionic: List[float] = []
    debye_nm: List[float] = []
    gamma_mean: List[float] = []
    gamma_na: List[float] = []
    gamma_cl: List[float] = []
    cond_molar: List[float] = []
    cond_spec: List[float] = []
    absorb_mean: List[float] = []
    braid_len: List[float] = []
    braid_writhe: List[float] = []
    dirac_gap: List[float] = []
    laplacian_gap: List[float] = []

    for i, r in enumerate(history):
        steps.append(int(r.get("step", i)))
        energy.append(_f(r.get("energy")))
        el = r.get("electrolyte") or {}
        ionic.append(_f(el.get("ionic_strength_M", r.get("ionic_strength_live_M"))))
        dnm = el.get("debye_length_nm")
        if dnm is not None and dnm != float("inf"):
            debye_nm.append(_f(dnm))
        else:
            debye_nm.append(float("nan"))
        gamma_mean.append(_f(el.get("gamma_mean"), 1.0))
        gamma_na.append(_f(el.get("gamma_na"), float("nan")))
        gamma_cl.append(_f(el.get("gamma_cl"), float("nan")))
        cond_molar.append(_f(el.get("molar_conductivity_s_cm2_mol"), float("nan")))
        cond_spec.append(_f(el.get("conductivity_s_cm_proxy"), float("nan")))
        ab = r.get("absorbance")
        if ab is not None:
            absorb_mean.append(_f(ab))
        else:
            spec = r.get("absorbance_spectrum") or {}
            if spec:
                absorb_mean.append(sum(_f(v) for v in spec.values()) / max(1, len(spec)))
            else:
                absorb_mean.append(float("nan"))
        braid_len.append(_f(r.get("braid_reduced_length")))
        braid_writhe.append(_f(r.get("braid_writhe")))
        dirac_gap.append(_f(r.get("dirac_gap")))
        laplacian_gap.append(_f(r.get("laplacian_gap")))

    return {
        "steps": steps,
        "energy": energy,
        "ionic_strength_M": ionic,
        "debye_length_nm": debye_nm,
        "gamma_mean": gamma_mean,
        "gamma_na": gamma_na,
        "gamma_cl": gamma_cl,
        "molar_conductivity": cond_molar,
        "conductivity_s_cm": cond_spec,
        "absorbance": absorb_mean,
        "braid_reduced_length": braid_len,
        "braid_writhe": braid_writhe,
        "dirac_gap": dirac_gap,
        "laplacian_gap": laplacian_gap,
        "n": n,
    }


# -----------------------------------------------------------------------------
# Matplotlib setup
# -----------------------------------------------------------------------------


def apply_publication_style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#fafafa",
            "figure.facecolor": "white",
        }
    )
    return plt


# -----------------------------------------------------------------------------
# Single-run plots
# -----------------------------------------------------------------------------


@dataclass
class PipelineResult:
    run_dir: Path
    figures_dir: Path
    written_png: List[str] = field(default_factory=list)
    written_pdf: List[str] = field(default_factory=list)


def _savefig(fig: Any, plt_module: Any, path: Path, pdf: bool) -> List[str]:
    out: List[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    out.append(str(path))
    if pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        out.append(str(pdf_path))
    plt_module.close(fig)
    return out


def generate_single_run(
    run_dir: str | Path,
    *,
    pdf: bool = False,
    prefix: str = "",
) -> PipelineResult:
    """Generate standard figures under ``run_dir/figures/``."""
    run_dir = Path(run_dir).resolve()
    history = load_history_efficient(run_dir)
    if not history:
        return PipelineResult(run_dir=run_dir, figures_dir=run_dir / "figures", written_png=[])

    s = extract_series(history)
    plt = apply_publication_style()
    fig_dir = run_dir / "figures"
    written: List[str] = []

    steps = s["steps"]
    label_tag = f"{prefix}: " if prefix else ""

    # energy.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["energy"], color="#1d4ed8", lw=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Free energy (simulator units)")
    ax.set_title(f"{label_tag}Total energy vs step")
    written += _savefig(fig, plt, fig_dir / "energy.png", pdf)

    # ionic_strength.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["ionic_strength_M"], color="#0f766e", lw=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Ionic strength $I$ (mol L$^{-1}$)")
    ax.set_title(f"{label_tag}Ionic strength")
    written += _savefig(fig, plt, fig_dir / "ionic_strength.png", pdf)

    # debye_length.png (explicit file per requirements-style naming — also ionic companion)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["debye_length_nm"], color="#b45309", lw=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Debye length $\lambda_D$ (nm)")
    ax.set_title(f"{label_tag}Debye screening length")
    written += _savefig(fig, plt, fig_dir / "debye_length.png", pdf)

    # gamma.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["gamma_mean"], color="#7c3aed", lw=1.8, label=r"$\gamma_{\mathrm{mean}}$")
    if any(x == x for x in s["gamma_na"]):  # any non-NaN
        ax.plot(steps, s["gamma_na"], color="#ca8a04", lw=1.4, alpha=0.85, label=r"$\gamma_{\mathrm{Na}^+}$")
    if any(x == x for x in s["gamma_cl"]):
        ax.plot(steps, s["gamma_cl"], color="#15803d", lw=1.4, alpha=0.85, label=r"$\gamma_{\mathrm{Cl}^-}$")
    ax.set_xlabel("Step")
    ax.set_ylabel("Activity coefficient (–)")
    ax.set_title(f"{label_tag}Activity coefficients")
    ax.legend(loc="best", framealpha=0.95)
    written += _savefig(fig, plt, fig_dir / "gamma.png", pdf)

    # conductivity.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if any(x == x for x in s["molar_conductivity"]) or any(x == x for x in s["conductivity_s_cm"]):
        ax.plot(
            steps,
            s["molar_conductivity"],
            color="#0369a1",
            lw=1.8,
            label=r"$\Lambda_m$ (S cm$^2$ mol$^{-1}$)",
        )
        ax.plot(
            steps,
            s["conductivity_s_cm"],
            color="#be123c",
            lw=1.6,
            alpha=0.9,
            label=r"$\kappa$ proxy (S cm$^{-1}$)",
        )
        ax.legend(loc="best", framealpha=0.95)
    else:
        ax.text(0.5, 0.5, "No Onsager metrics (--use_onsager)", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Step")
    ax.set_ylabel("Conductivity / molar conductivity (proxy)")
    ax.set_title(f"{label_tag}Conductivity (Onsager/Kohlrausch proxy)")
    written += _savefig(fig, plt, fig_dir / "conductivity.png", pdf)

    # absorbance.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if any(x == x for x in s["absorbance"]):
        ax.plot(steps, s["absorbance"], color="#db2777", lw=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Absorbance proxy (–)")
    ax.set_title(f"{label_tag}Absorbance (mean / spectrum aggregate)")
    written += _savefig(fig, plt, fig_dir / "absorbance.png", pdf)

    # braid.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["braid_reduced_length"], color="#4338ca", lw=1.8, label="Reduced length")
    ax.plot(steps, s["braid_writhe"], color="#ea580c", lw=1.5, alpha=0.9, label="Writhe")
    ax.set_xlabel("Step")
    ax.set_ylabel("Braid metric (–)")
    ax.set_title(f"{label_tag}Braid complexity")
    ax.legend(loc="best", framealpha=0.95)
    written += _savefig(fig, plt, fig_dir / "braid.png", pdf)

    # spectral.png
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, s["dirac_gap"], color="#0e7490", lw=1.8, label="Dirac gap")
    ax.plot(steps, s["laplacian_gap"], color="#a21caf", lw=1.6, alpha=0.9, label="Laplacian gap")
    ax.set_xlabel("Step")
    ax.set_ylabel("Spectral gap (simulator units)")
    ax.set_title(f"{label_tag}Spectral gaps (NCG)")
    ax.legend(loc="best", framealpha=0.95)
    written += _savefig(fig, plt, fig_dir / "spectral.png", pdf)

    png_files = [w for w in written if w.endswith(".png")]
    pdf_files = [w for w in written if w.endswith(".pdf")]
    (fig_dir / "pipeline_manifest.json").write_text(
        json.dumps({"png": png_files, "pdf": pdf_files}, indent=2),
        encoding="utf-8",
    )
    return PipelineResult(run_dir=run_dir, figures_dir=fig_dir, written_png=png_files, written_pdf=pdf_files)


# -----------------------------------------------------------------------------
# Comparison overlays
# -----------------------------------------------------------------------------


def generate_compare(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    pdf: bool = False,
) -> List[str]:
    """Overlay multiple runs; saves under ``out_dir/figures/``."""
    run_dirs = [Path(d).resolve() for d in run_dirs]
    out_dir = Path(out_dir).resolve()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    series_list: List[Tuple[str, Dict[str, Any]]] = []
    for rd in run_dirs:
        hist = load_history_efficient(rd)
        if not hist:
            continue
        label = rd.name
        series_list.append((label, extract_series(hist)))
    if len(series_list) < 2:
        raise ValueError("Need at least two non-empty runs for --compare")

    plt = apply_publication_style()
    written: List[str] = []

    def overlay(key_y: str, fname: str, ylabel: str, title: str):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        cmap = ["#1d4ed8", "#b45309", "#15803d", "#7c3aed", "#be123c"]
        for i, (label, s) in enumerate(series_list):
            c = cmap[i % len(cmap)]
            y = s[key_y]
            ax.plot(s["steps"], y, lw=1.6, color=c, label=label, alpha=0.95)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)
        written.extend(_savefig(fig, plt, fig_dir / fname, pdf))

    overlay("energy", "compare_energy.png", "Energy (sim. units)", "Comparison: energy")
    overlay("ionic_strength_M", "compare_ionic_strength.png", r"$I$ (mol L$^{-1}$)", "Comparison: ionic strength")
    overlay("gamma_mean", "compare_gamma.png", r"$\gamma_{\mathrm{mean}}$", "Comparison: mean activity coefficient")
    overlay("dirac_gap", "compare_dirac_gap.png", "Dirac gap", "Comparison: Dirac spectral gap")
    overlay(
        "laplacian_gap",
        "compare_laplacian_gap.png",
        "Laplacian gap",
        "Comparison: Laplacian spectral gap",
    )

    (fig_dir / "compare_manifest.json").write_text(json.dumps({"figures": written}, indent=2), encoding="utf-8")
    return written


# -----------------------------------------------------------------------------
# LaTeX snippet
# -----------------------------------------------------------------------------


LATEX_TEMPLATE = r"""% Auto-generated by srg_hbond.figures_pipeline
% Place \input{figures} or copy blocks into your manuscript preamble as needed.
\usepackage{graphicx}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{energy.png}
  \caption{Total free energy vs.\ simulation step.}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{ionic_strength.png}
  \caption{Ionic strength $I$ (mol\,L$^{-1}$).}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{debye_length.png}
  \caption{Debye screening length $\lambda_D$ (nm).}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{gamma.png}
  \caption{Mean and ion activity coefficients (model-dependent).}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{conductivity.png}
  \caption{Molar and specific conductivity proxies (Onsager/Kohlrausch style).}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{absorbance.png}
  \caption{Absorbance proxy vs.\ step.}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{braid.png}
  \caption{Braid reduced length and writhe.}
\end{figure}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.92\linewidth]{spectral.png}
  \caption{Dirac and Laplacian spectral gaps (NCG layer).}
\end{figure}
"""


def write_latex_snippet(figures_dir: Path) -> Path:
    tex_path = figures_dir / "figures.tex"
    tex_path.write_text(LATEX_TEMPLATE, encoding="utf-8")
    return tex_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="SRG v6.1 publication-style figures from run directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_dir", type=str, default=None, help="Single experiment directory under runs/")
    ap.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Two or more run directories to overlay",
    )
    ap.add_argument(
        "--compare_out",
        type=str,
        default=None,
        help="Output directory for comparison figures (default: runs/compare_<first>)",
    )
    ap.add_argument(
        "--all",
        type=str,
        default=None,
        dest="batch_root",
        metavar="ROOT",
        help="Process every subdirectory of ROOT that contains history data (e.g. runs/)",
    )
    ap.add_argument("--pdf", action="store_true", help="Also write vector PDF for each figure")
    ap.add_argument("--latex", action="store_true", help="Write figures.tex next to PNGs (single-run mode)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.compare:
        if len(args.compare) < 2:
            ap.error("--compare requires at least two run directories")
        first = Path(args.compare[0]).resolve().name
        out = Path(args.compare_out or f"runs/compare_{first}")
        paths = generate_compare(args.compare, out, pdf=args.pdf)
        print(f"Wrote {len(paths)} file(s) under {out / 'figures'}")
        if args.latex:
            write_latex_snippet(out / "figures")
            print(f"Wrote {out / 'figures' / 'figures.tex'}")
        return 0

    if args.batch_root:
        root = Path(args.batch_root)
        dirs = discover_run_directories(root)
        if not dirs:
            print(f"No experiment directories found under {root}")
            return 1
        for d in dirs:
            res = generate_single_run(d, pdf=args.pdf)
            print(f"{d.name}: {len(res.written_png)} PNG(s) -> {res.figures_dir}")
            if args.latex:
                tp = write_latex_snippet(res.figures_dir)
                print(f"  LaTeX: {tp}")
        return 0

    if args.run_dir:
        res = generate_single_run(args.run_dir, pdf=args.pdf)
        print(f"Wrote {len(res.written_png)} PNG(s) -> {res.figures_dir}")
        if args.latex:
            tp = write_latex_snippet(res.figures_dir)
            print(f"LaTeX snippet: {tp}")
        return 0

    ap.error("Provide --run_dir, --compare, or --all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
