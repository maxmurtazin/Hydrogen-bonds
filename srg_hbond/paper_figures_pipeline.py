"""
Paper-ready figures, LaTeX bundles, and summary tables for SRG Chemical Computing Platform v6.1.

Examples::

    python3 -m srg_hbond.paper_figures_pipeline \\
        --run_dir runs/water_nacl --paper_dir paper_outputs/water_nacl

    python3 -m srg_hbond.paper_figures_pipeline \\
        --compare runs/a runs/b --paper_dir paper_outputs/comparison

    python3 -m srg_hbond.paper_figures_pipeline \\
        --all runs --paper_dir paper_outputs/all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Warnings collector (print at end)
# -----------------------------------------------------------------------------

_LOG: List[str] = []


def _warn(msg: str) -> None:
    _LOG.append(msg)
    print(f"[paper_figures_pipeline] WARNING: {msg}")


def _flush_warnings() -> None:
    if _LOG:
        print(f"[paper_figures_pipeline] {len(_LOG)} warning(s) logged.")


# -----------------------------------------------------------------------------
# Loading & flattening
# -----------------------------------------------------------------------------


def flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Promote ``electrolyte`` dict to top-level ``electrolyte_*`` keys."""
    r = dict(record)
    el = r.pop("electrolyte", None)
    if isinstance(el, dict):
        for k, v in el.items():
            r[f"electrolyte_{k}"] = v
    return r


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "frames" in raw:
        raw = raw["frames"]
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    return [flatten_record(x) for x in raw]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(flatten_record(json.loads(line)))
    return rows


def load_run_data(run_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load trajectory with priority:
    history.json → frames_live.json → metrics_bundle.jsonl → frames.json
    """
    run_dir = Path(run_dir).resolve()
    candidates: List[Tuple[str, Path]] = [
        ("json", run_dir / "history.json"),
        ("json", run_dir / "frames_live.json"),
        ("jsonl", run_dir / "metrics_bundle.jsonl"),
        ("json", run_dir / "frames.json"),
    ]
    for kind, p in candidates:
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            if kind == "jsonl":
                return _load_jsonl(p)
            return _load_json_list(p)
        except (json.JSONDecodeError, ValueError) as e:
            _warn(f"Skip {p}: {e}")
            continue
    raise FileNotFoundError(
        f"No loadable history in {run_dir} (tried history.json, frames_live.json, "
        "metrics_bundle.jsonl, frames.json)"
    )


def discover_runs(root: Path) -> List[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    found: List[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            load_run_data(d)
        except FileNotFoundError:
            continue
        found.append(d)
    return found


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    n = len(values)
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - window + 1)
        chunk = [values[j] for j in range(lo, i + 1) if not math.isnan(values[j])]
        out.append(statistics.mean(chunk) if chunk else float("nan"))
    return out


# -----------------------------------------------------------------------------
# Series extraction (tolerant)
# -----------------------------------------------------------------------------


def extract_series(history: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    steps: List[int] = []
    energy: List[float] = []
    reward: List[float] = []
    absorb: List[float] = []
    ionic: List[float] = []
    debye: List[float] = []
    g_na: List[float] = []
    g_cl: List[float] = []
    g_mean: List[float] = []
    cond: List[float] = []
    dirac: List[float] = []
    lap: List[float] = []
    ncg_s: List[float] = []
    braid_len: List[float] = []
    braid_ent: List[float] = []
    braid_wr: List[float] = []
    elapsed: List[float] = []
    eta: List[float] = []
    sps: List[float] = []

    for i, r in enumerate(history):
        r = flatten_record(dict(r))
        steps.append(int(r.get("step", i)))
        energy.append(_f(r.get("energy")))
        reward.append(_f(r.get("reward")))
        ab = r.get("absorbance")
        if ab is None:
            spec = r.get("absorbance_spectrum") or {}
            if isinstance(spec, dict) and spec:
                ab = sum(_f(v) for v in spec.values()) / len(spec)
            else:
                ab = float("nan")
        absorb.append(_f(ab))
        el = r.get("electrolyte_ionic_strength_M")
        if el is None:
            el = (r.get("electrolyte") or {}).get("ionic_strength_M") if isinstance(
                r.get("electrolyte"), dict
            ) else None
        if el is None:
            el = r.get("ionic_strength_live_M")
        ionic.append(_f(el))
        dnm = r.get("electrolyte_debye_length_nm")
        if dnm is None:
            dnm = (r.get("electrolyte") or {}).get("debye_length_nm") if isinstance(
                r.get("electrolyte"), dict
            ) else None
        if dnm is not None and dnm != float("inf"):
            debye.append(_f(dnm))
        else:
            debye.append(float("nan"))

        g_na.append(_f(r.get("electrolyte_gamma_na") or r.get("gamma_na")))
        g_cl.append(_f(r.get("electrolyte_gamma_cl") or r.get("gamma_cl")))
        gm = r.get("electrolyte_gamma_mean") or r.get("gamma_mean")
        if gm is None and isinstance(r.get("electrolyte"), dict):
            gm = r["electrolyte"].get("gamma_mean")
        g_mean.append(_f(gm, 1.0))

        ck = r.get("electrolyte_conductivity_s_cm_proxy")
        if ck is None and isinstance(r.get("electrolyte"), dict):
            ck = r["electrolyte"].get("conductivity_s_cm_proxy")
        cond.append(_f(ck))

        dirac.append(_f(r.get("dirac_gap")))
        lap.append(_f(r.get("laplacian_gap")))
        ncg_s.append(_f(r.get("ncg_smoothness")))
        braid_len.append(_f(r.get("braid_reduced_length")))
        be = r.get("braid_entropy") or r.get("braid_generator_entropy")
        braid_ent.append(_f(be))
        braid_wr.append(_f(r.get("braid_writhe")))
        elapsed.append(_f(r.get("elapsed_s")))
        eta.append(_f(r.get("eta_s")))
        sps.append(_f(r.get("steps_per_s")))

    return {
        "steps": steps,
        "energy": energy,
        "reward": reward,
        "absorbance": absorb,
        "ionic_strength_M": ionic,
        "debye_length_nm": debye,
        "gamma_na": g_na,
        "gamma_cl": g_cl,
        "gamma_mean": g_mean,
        "conductivity_proxy": cond,
        "dirac_gap": dirac,
        "laplacian_gap": lap,
        "ncg_smoothness": ncg_s,
        "braid_reduced_length": braid_len,
        "braid_entropy": braid_ent,
        "braid_writhe": braid_wr,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "steps_per_s": sps,
    }


def maybe_smooth(
    y: List[float], window: int, plot_raw_when_smooth: bool = True
) -> Tuple[List[float], Optional[List[float]]]:
    raw = list(y)
    if window <= 1:
        return raw, None
    sm = moving_average(y, window)
    return sm, raw if plot_raw_when_smooth else None


# -----------------------------------------------------------------------------
# Summary metrics
# -----------------------------------------------------------------------------


def compute_summary_metrics(series: Mapping[str, List[float]]) -> Dict[str, Any]:
    def fin(key: str) -> float:
        v = series.get(key) or []
        if not v:
            return float("nan")
        x = v[-1]
        return x if not math.isnan(x) else float("nan")

    def ini(key: str) -> float:
        v = series.get(key) or []
        if not v:
            return float("nan")
        x = v[0]
        return x if not math.isnan(x) else float("nan")

    def minv(key: str) -> float:
        v = [x for x in (series.get(key) or []) if not math.isnan(x)]
        return min(v) if v else float("nan")

    eng = series.get("energy") or []
    n_steps = len(eng)
    return {
        "n_steps": n_steps,
        "initial_energy": ini("energy"),
        "final_energy": fin("energy"),
        "min_energy": minv("energy"),
        "final_reward": fin("reward"),
        "final_absorbance": fin("absorbance"),
        "final_ionic_strength": fin("ionic_strength_M"),
        "final_debye_length_nm": fin("debye_length_nm"),
        "final_gamma_na": fin("gamma_na"),
        "final_gamma_cl": fin("gamma_cl"),
        "final_gamma_mean": fin("gamma_mean"),
        "final_conductivity_proxy": fin("conductivity_proxy"),
        "final_dirac_gap": fin("dirac_gap"),
        "final_laplacian_gap": fin("laplacian_gap"),
        "final_ncg_smoothness": fin("ncg_smoothness"),
        "final_braid_reduced_length": fin("braid_reduced_length"),
        "final_braid_entropy": fin("braid_entropy"),
        "final_braid_writhe": fin("braid_writhe"),
    }


# -----------------------------------------------------------------------------
# Matplotlib
# -----------------------------------------------------------------------------


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#fafafa",
        }
    )
    return plt


def plot_metric(
    plt,
    steps: List[int],
    y: List[float],
    ylabel: str,
    title: str,
    *,
    y_raw: Optional[List[float]] = None,
    color: str = "#1e40af",
) -> Any:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    if y_raw is not None:
        ax.plot(steps, y_raw, color=color, lw=0.9, alpha=0.35, label="Raw")
        ax.plot(steps, y, color=color, lw=1.8, label="Smoothed")
        ax.legend(loc="best", framealpha=0.95)
    else:
        ax.plot(steps, y, color=color, lw=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_overlay(
    plt,
    runs: Sequence[Tuple[str, Dict[str, List[float]], List[int]]],
    key: str,
    ylabel: str,
    title: str,
) -> Any:
    cmap = ["#1d4ed8", "#b45309", "#15803d", "#7c3aed", "#be123c", "#0e7490"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (label, s, steps) in enumerate(runs):
        y = s.get(key) or []
        if not y:
            _warn(f"Missing series '{key}' for run {label}")
            continue
        ax.plot(steps, y, lw=1.5, color=cmap[i % len(cmap)], label=label, alpha=0.95)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return fig


def _save_fig(fig: Any, plt, path_base: Path) -> Tuple[str, str]:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    png = path_base.with_suffix(".png")
    pdf = path_base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)


# -----------------------------------------------------------------------------
# LaTeX helpers (plain text vs math)
# -----------------------------------------------------------------------------

# Prevent accidental closure of LaTeX verbatim environments when dumping JSON.
_VERBATIM_GUARD = "<<<SRG_VERBATIM_GUARD>>>"

_RE_UNSAFE_PLAIN = re.compile(r"[%&$#_{}~^\\]")


def tex_escape(s: Any) -> str:
    """
    Escape arbitrary plain text for LaTeX text mode (not inside $...$).
    Escapes: \\ % & # $ { } _ ^ ~
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    out: List[str] = []
    for c in s:
        if c == "\\":
            out.append(r"\textbackslash{}")
        elif c == "{":
            out.append(r"\{")
        elif c == "}":
            out.append(r"\}")
        elif c == "$":
            out.append(r"\$")
        elif c == "&":
            out.append(r"\&")
        elif c == "#":
            out.append(r"\#")
        elif c == "%":
            out.append(r"\%")
        elif c == "_":
            out.append(r"\_")
        elif c == "^":
            out.append(r"\textasciicircum{}")
        elif c == "~":
            out.append(r"\textasciitilde{}")
        else:
            out.append(c)
    return "".join(out)


def warn_unsafe_plain_text(s: str, context: str = "") -> None:
    """Optional validation: warn if string looks like unescaped LaTeX specials."""
    if s is None or not isinstance(s, str):
        return
    if _RE_UNSAFE_PLAIN.search(s):
        _warn(f"Possibly unsafe characters in LaTeX plain text{(': ' + context) if context else ''}")


def latex_math_num(x: Any, nd: int = 4) -> str:
    """Format a scalar for use *inside* math mode only (digits, minus, e)."""
    if x is None:
        return r"\mathrm{NA}"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return r"\mathrm{NA}"
    if math.isnan(v) or math.isinf(v):
        return r"\mathrm{NA}"
    s = f"{v:.{nd}g}"
    if any(c in s for c in "_^%$&"):
        return r"\mathrm{NA}"
    return s


def latex_table_cell(x: Any, nd: int = 4) -> str:
    """Format a table cell in text mode: numbers formatted, strings escaped."""
    if x is None:
        return "---"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "---"
        return f"{x:.{nd}g}"
    return tex_escape(str(x))


def sanitize_verbatim_body(text: str) -> str:
    """Prevent JSON/text from closing \\begin{verbatim} prematurely."""
    return text.replace(r"\end{verbatim}", _VERBATIM_GUARD)


def _latex_num(x: float, nd: int = 4) -> str:
    """Legacy numeric formatter for text mode tables (ASCII-only output)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "---"
    return f"{x:.{nd}g}"


def tex_escape_graphics_filename(path: str) -> str:
    """Underscores in \\includegraphics{{...}} trigger subscripts; escape them."""
    return path.replace("_", r"\_")


def latex_safe_label(s: str) -> str:
    """\\label{{...}} cannot contain raw underscores (math subscript). Use hyphens."""
    return s.replace("_", "-")


def write_latex_figure_blocks(
    figures_rel: Sequence[str],
    captions: Mapping[str, str],
    labels: Mapping[str, str],
) -> str:
    lines: List[str] = []
    for stem in figures_rel:
        key = Path(stem).stem
        cap = tex_escape(captions.get(key, ""))
        lab = latex_safe_label(labels.get(key, f"fig:{key}"))
        lines.append("")
        lines.append("\\begin{figure}[t]")
        lines.append("  \\centering")
        safe_name = tex_escape_graphics_filename(stem)
        lines.append(f"  \\includegraphics[width=0.9\\linewidth]{{{safe_name}}}")
        lines.append(f"  \\caption{{{cap}}}")
        lines.append(f"  \\label{{{lab}}}")
        lines.append("\\end{figure}")
    return "\n".join(lines)


def write_results_summary_tex(path: Path, summary: Mapping[str, Any], run_name: str) -> None:
    """Narrative paragraph: plain text escaped; numbers only inside math $...$."""
    n = int(summary.get("n_steps") or 0)
    rn = tex_escape(run_name)
    ei = latex_math_num(summary.get("initial_energy"))
    ef = latex_math_num(summary.get("final_energy"))
    em = latex_math_num(summary.get("min_energy"))
    I = latex_math_num(summary.get("final_ionic_strength"))
    ld = latex_math_num(summary.get("final_debye_length_nm"))
    dg = latex_math_num(summary.get("final_dirac_gap"))
    br = latex_math_num(summary.get("final_braid_reduced_length"))

    para = (
        f"Across {n} simulation steps ({rn}), the coarse-grained free energy "
        f"evolved from ${ei}$ to ${ef}$ (minimum ${em}$). "
        f"The electrolyte layer reports ionic strength $I = {I}$ (mol per liter) "
        f"and Debye screening length $\\lambda_D = {ld}$~nm. "
        f"Spectral indicators yield a Dirac gap of ${dg}$ (simulator units). "
        f"Braid complexity (reduced word length) reaches ${br}$. "
        "These metrics are qualitative graph-level proxies; quantitative agreement with experiment is not implied."
    )
    body = "% Auto-generated by paper_figures_pipeline.py\n\\paragraph{Results summary.}\n" + para + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def write_metrics_table_tex(path: Path, summary: Mapping[str, Any]) -> None:
    """Table labels are ASCII-only; dynamic values use latex_table_cell."""
    rows: List[Tuple[str, str]] = [
        ("Initial energy", "initial_energy"),
        ("Final energy", "final_energy"),
        ("Minimum energy", "min_energy"),
        ("Final absorbance proxy", "final_absorbance"),
        ("Final ionic strength (mol per liter)", "final_ionic_strength"),
        ("Final Debye length (nm)", "final_debye_length_nm"),
        ("Final gamma mean", "final_gamma_mean"),
        ("Final conductivity proxy (S per cm)", "final_conductivity_proxy"),
        ("Final Dirac gap", "final_dirac_gap"),
        ("Final braid reduced length", "final_braid_reduced_length"),
    ]
    lines = [
        "% Auto-generated",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Summary metrics from the SRG simulation run.}",
        "  \\label{tab:srg_metrics}",
        "  \\begin{tabular}{lr}",
        "    \\toprule",
        "    Quantity & Value \\\\",
        "    \\midrule",
    ]
    for label, key in rows:
        lines.append(f"    {label} & {latex_table_cell(summary.get(key))} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_captions_tex(path: Path) -> None:
    caps = {
        "fig_energy": "Total free energy versus simulation step.",
        "fig_reward": "Shaped reward per step (includes NCG/braid corrections when enabled).",
        "fig_absorbance": "Mean absorbance proxy versus step.",
        "fig_electrolyte": "Ionic strength and Debye screening length.",
        "fig_activity": "Mean and ion-specific activity coefficients (model-dependent).",
        "fig_conductivity": "Onsager/Kohlrausch conductivity proxies when available.",
        "fig_spectral": "Dirac gap, Laplacian gap, and NCG smoothness.",
        "fig_braid": "Braid reduced length, generator entropy, and writhe.",
        "fig_performance": "Wall-clock elapsed time, ETA estimate, and steps per second.",
        "fig_dashboard_summary": "Four-panel overview: energy, ionic strength, Dirac gap, braid length.",
        "fig_compare_energy": "Comparison of energy traces across runs.",
        "fig_compare_absorbance": "Comparison of absorbance proxies.",
        "fig_compare_ionic": "Comparison of ionic strength.",
        "fig_compare_conductivity": "Comparison of conductivity proxies.",
        "fig_compare_spectral": "Comparison of Dirac spectral gaps.",
        "fig_compare_braid": "Comparison of braid reduced lengths.",
        "fig_compare_bars": "Final-energy comparison across runs.",
    }
    lines = ["% Caption macros — \\caption{\\captionXXX}", ""]
    for k, v in sorted(caps.items()):
        macro = k.replace("fig_", "caption").title().replace("_", "")
        # simpler: \newcommand{\captionEnergy}{...}
        safe = "".join(x if x.isalnum() else "" for x in k.split("_")[-1])
        lines.append(f"\\newcommand{{\\Caption{safe.title()}}}{{{v}}}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "% Use e.g. \\caption{\\CaptionEnergy} if matching macros added.\n"
        + "\n".join(f"% {k}: {v}" for k, v in sorted(caps.items())),
        encoding="utf-8",
    )


def write_mini_report(
    path: Path,
    *,
    title: str,
    run_label: str,
    settings_lines: List[str],
    summary_tex_rel: str,
    table_tex_rel: str,
    figure_stems: Sequence[str],
) -> None:
    """mini_report.tex lives in report/; graphics in ../figures/. All user text escaped."""
    title_tex = tex_escape(title)
    settings_blob = sanitize_verbatim_body(chr(10).join(settings_lines))
    body = f"""% Auto-generated SRG mini report (pdflatex-safe)
\\documentclass[11pt]{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{textcomp}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{url}}
\\graphicspath{{{{../figures/}}}}

\\title{{{title_tex}}}
\\author{{SRG Chemical Computing Platform v6.1}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Experimental settings}}
Run identifier (escaped): \\texttt{{{tex_escape(run_label)}}}.
\\begin{{verbatim}}
{settings_blob}
\\end{{verbatim}}

\\section{{Summary metrics}}
\\input{{{table_tex_rel}}}

\\section{{Results narrative}}
\\input{{{summary_tex_rel}}}

\\section{{Figures}}
"""
    for stem in figure_stems:
        safe_stem = tex_escape_graphics_filename(stem)
        cap_base = tex_escape(Path(stem).stem)
        body += f"""
\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.92\\linewidth]{{{safe_stem}}}
  \\caption{{Figure: {cap_base}}}
\\end{{figure}}
"""
    body += """
\\section{Interpretation}
The traces summarize coarse-grained graph energetics, optional electrolyte corrections,
spectral indicators from the discrete Dirac operator, and braid statistics on Monte Carlo moves.

\\section{Limitations}
This MVP uses qualitative energy mappings and simplified DH/Davies/Pitzer proxies; it is not
a calibrated molecular simulation or experimental predictor.

\\end{document}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# -----------------------------------------------------------------------------
# Core pipeline — single run
# -----------------------------------------------------------------------------


def process_single(
    run_dir: Path,
    paper_root: Path,
    *,
    smooth_window: int = 1,
) -> Dict[str, Any]:
    global _LOG
    _LOG = []
    paper_root = paper_root.resolve()
    fig_dir = paper_root / "figures"
    tex_dir = paper_root / "tex"
    met_dir = paper_root / "metrics"
    rep_dir = paper_root / "report"
    for d in (fig_dir, tex_dir, met_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    history = load_run_data(run_dir)
    if not history:
        _warn(f"Empty history: {run_dir}")
    series = extract_series(history)
    sw = max(1, smooth_window)

    plt = _setup_mpl()
    written: List[str] = []

    def save(name: str, fig) -> None:
        nonlocal written
        p = fig_dir / name
        written.extend(_save_fig(fig, plt, p))

    # 1 Energy
    ye, ye_raw = maybe_smooth(series["energy"], sw)
    save(
        "fig_energy",
        plot_metric(
            plt,
            series["steps"],
            ye,
            "Free energy (arb.)",
            "Energy relaxation",
            y_raw=ye_raw,
        ),
    )

    # 2 Reward
    yr, yr_raw = maybe_smooth(series["reward"], sw)
    save(
        "fig_reward",
        plot_metric(
            plt, series["steps"], yr, "Reward (arb.)", "Reward trajectory", y_raw=yr_raw, color="#7c2d12"
        ),
    )

    # 3 Absorbance
    ya, ya_raw = maybe_smooth(series["absorbance"], sw)
    save(
        "fig_absorbance",
        plot_metric(
            plt,
            series["steps"],
            ya,
            "Absorbance proxy",
            "Absorbance",
            y_raw=ya_raw,
            color="#be185d",
        ),
    )

    # 4–5 Electrolyte panel
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
    yi, _ = maybe_smooth(series["ionic_strength_M"], sw)
    yd, _ = maybe_smooth(series["debye_length_nm"], sw)
    ax1.plot(series["steps"], yi, color="#0f766e", lw=1.6)
    ax1.set_ylabel(r"$I$ (mol L$^{-1}$)")
    ax2.plot(series["steps"], yd, color="#c2410c", lw=1.6)
    ax2.set_xlabel("Step")
    ax2.set_ylabel(r"$\lambda_D$ (nm)")
    fig.suptitle("Electrolyte metrics")
    fig.tight_layout()
    save("fig_electrolyte", fig)

    # 6 Activity
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for lab, key, c in [
        (r"$\gamma_{\mathrm{mean}}$", "gamma_mean", "#6d28d9"),
        (r"$\gamma_{\mathrm{Na}^+}$", "gamma_na", "#ca8a04"),
        (r"$\gamma_{\mathrm{Cl}^-}$", "gamma_cl", "#16a34a"),
    ]:
        v = series[key]
        if any(not math.isnan(x) for x in v):
            ax.plot(series["steps"], v, label=lab, lw=1.4, color=c)
        else:
            _warn(f"Activity series empty: {key}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Activity coefficient")
    ax.legend(loc="best")
    ax.set_title("Activity coefficients")
    fig.tight_layout()
    save("fig_activity", fig)

    # 7 Conductivity
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    yc, _ = maybe_smooth(series["conductivity_proxy"], sw)
    if any(not math.isnan(x) for x in yc):
        ax.plot(series["steps"], yc, color="#0369a1", lw=1.8)
    else:
        ax.text(0.5, 0.5, "No conductivity data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\kappa$ proxy (S cm$^{-1}$)")
    ax.set_title("Conductivity proxy")
    fig.tight_layout()
    save("fig_conductivity", fig)

    # 8 Spectral
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(series["steps"], series["dirac_gap"], label="Dirac gap", lw=1.5)
    ax.plot(series["steps"], series["laplacian_gap"], label="Laplacian gap", lw=1.4, alpha=0.85)
    ax.plot(series["steps"], series["ncg_smoothness"], label="NCG smoothness", lw=1.2, alpha=0.8)
    ax.set_xlabel("Step")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Spectral / NCG metrics")
    fig.tight_layout()
    save("fig_spectral", fig)

    # 9 Braid
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(series["steps"], series["braid_reduced_length"], label="Reduced length", lw=1.5)
    if any(not math.isnan(x) for x in series["braid_entropy"]):
        ax.plot(series["steps"], series["braid_entropy"], label="Generator entropy", lw=1.3, alpha=0.9)
    else:
        _warn("braid_entropy / braid_generator_entropy not present in frames")
    ax.plot(series["steps"], series["braid_writhe"], label="Writhe", lw=1.2, alpha=0.85)
    ax.set_xlabel("Step")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Braid metrics")
    fig.tight_layout()
    save("fig_braid", fig)

    # 10 Performance
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    if any(not math.isnan(x) for x in series["elapsed_s"]):
        ax.plot(series["steps"], series["elapsed_s"], label="Elapsed (s)", lw=1.4)
    if any(not math.isnan(x) for x in series["eta_s"]):
        ax.plot(series["steps"], series["eta_s"], label="ETA (s)", lw=1.2, alpha=0.85)
    if any(not math.isnan(x) for x in series["steps_per_s"]):
        ax.plot(series["steps"], series["steps_per_s"], label="Steps/s", lw=1.2, alpha=0.85)
    if not any(
        any(not math.isnan(x) for x in series[k])
        for k in ("elapsed_s", "eta_s", "steps_per_s")
    ):
        ax.text(0.5, 0.5, "No timing fields in frames", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Step")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Performance")
    fig.tight_layout()
    save("fig_performance", fig)

    # Dashboard summary 2x2
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))
    axes[0, 0].plot(series["steps"], series["energy"], color="#1e40af", lw=1.4)
    axes[0, 0].set_title("Energy")
    axes[0, 1].plot(series["steps"], yi, color="#0f766e", lw=1.4)
    axes[0, 1].set_title(r"Ionic strength $I$")
    axes[1, 0].plot(series["steps"], series["dirac_gap"], color="#0e7490", lw=1.4)
    axes[1, 0].set_title("Dirac gap")
    axes[1, 1].plot(series["steps"], series["braid_reduced_length"], color="#4338ca", lw=1.4)
    axes[1, 1].set_title("Braid reduced length")
    for ax in axes.flat:
        ax.set_xlabel("Step")
    fig.suptitle("Dashboard summary")
    fig.tight_layout()
    save("fig_dashboard_summary", fig)

    summary = compute_summary_metrics(series)
    summary["run_dir"] = str(run_dir)
    summary["run_name"] = run_dir.name

    met_dir.mkdir(parents=True, exist_ok=True)
    (met_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(met_dir / "summary_metrics.csv", summary)

    caps = {
        "fig_energy": "Energy relaxation during the SRG simulation.",
        "fig_reward": "Reward trajectory including spectral and braid shaping.",
        "fig_absorbance": "Absorbance proxy versus step.",
        "fig_electrolyte": "Ionic strength and Debye length.",
        "fig_activity": "Activity coefficients from the selected electrolyte model.",
        "fig_conductivity": "Specific conductivity proxy (Onsager/Kohlrausch style).",
        "fig_spectral": "Spectral gaps and NCG smoothness.",
        "fig_braid": "Braid complexity metrics.",
        "fig_performance": "Timing and throughput.",
        "fig_dashboard_summary": "Four-panel overview of key observables.",
    }
    labs = {k: latex_safe_label(f"fig:{k.replace('fig_', '')}") for k in caps}

    # Build figures.tex with correct stems
    stems = [
        "fig_energy",
        "fig_reward",
        "fig_absorbance",
        "fig_electrolyte",
        "fig_activity",
        "fig_conductivity",
        "fig_spectral",
        "fig_braid",
        "fig_performance",
        "fig_dashboard_summary",
    ]
    ftex_lines = [
        "% Auto-generated — in master document use: \\graphicspath{{paper_outputs/<run>/figures/}}",
        "",
    ]
    for stem in stems:
        cap = tex_escape(caps.get(stem, ""))
        lab = latex_safe_label(labs.get(stem, f"fig:{stem.replace('fig_', '')}"))
        gname = tex_escape_graphics_filename(f"{stem}.pdf")
        ftex_lines.extend(
            [
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.9\\linewidth]{{{gname}}}",
                f"  \\caption{{{cap}}}",
                f"  \\label{{{lab}}}",
                "\\end{figure}",
                "",
            ]
        )
    (tex_dir / "figures.tex").write_text("\n".join(ftex_lines), encoding="utf-8")

    write_captions_tex(tex_dir / "captions.tex")
    write_metrics_table_tex(tex_dir / "table_metrics.tex", summary)
    write_results_summary_tex(tex_dir / "results_summary.tex", summary, run_dir.name)

    settings_lines = _gather_settings_lines(run_dir)
    write_mini_report(
        rep_dir / "mini_report.tex",
        title=f"SRG run: {run_dir.name}",
        run_label=run_dir.name,
        settings_lines=settings_lines,
        summary_tex_rel="../tex/results_summary",
        table_tex_rel="../tex/table_metrics",
        figure_stems=[f"{s}.pdf" for s in stems],
    )

    _flush_warnings()
    return {"paper_root": str(paper_root), "summary": summary, "figures": written}


def _gather_settings_lines(run_dir: Path) -> List[str]:
    lines: List[str] = []
    for name in (
        "electrolyte_params_report.json",
        "electrolyte_settings.json",
        "physical_params_report.json",
    ):
        p = run_dir / name
        if p.exists():
            lines.append(f"=== {name} ===")
            try:
                lines.append(json.dumps(json.loads(p.read_text()), indent=2))
            except json.JSONDecodeError:
                lines.append(p.read_text()[:2000])
    if not lines:
        lines.append("(no settings JSON found in run directory)")
    return lines


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "value"])
        for k, v in sorted(summary.items()):
            w.writerow([k, v])


# -----------------------------------------------------------------------------
# Comparison mode
# -----------------------------------------------------------------------------


def process_compare(
    run_dirs: Sequence[Path],
    paper_root: Path,
    smooth_window: int = 1,
) -> Dict[str, Any]:
    global _LOG
    _LOG = []
    paper_root = paper_root.resolve()
    fig_dir = paper_root / "figures"
    tex_dir = paper_root / "tex"
    met_dir = paper_root / "metrics"
    rep_dir = paper_root / "report"
    for d in (fig_dir, tex_dir, met_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)

    bundled: List[Tuple[str, Dict[str, List[float]], List[int]]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    plt = _setup_mpl()

    for rd in run_dirs:
        rd = Path(rd).resolve()
        try:
            hist = load_run_data(rd)
        except FileNotFoundError as e:
            _warn(str(e))
            continue
        s = extract_series(hist)
        sw = max(1, smooth_window)
        if sw > 1:
            for key in list(s.keys()):
                if key != "steps" and isinstance(s[key], list):
                    s[key], _ = maybe_smooth(s[key], sw)
        bundled.append((rd.name, s, s["steps"]))
        summaries[rd.name] = compute_summary_metrics(s)

    if len(bundled) < 2:
        raise SystemExit("Compare mode needs at least two valid runs.")

    def cs(name: str, fig) -> None:
        _save_fig(fig, plt, fig_dir / name)

    cs(
        "fig_compare_energy",
        plot_overlay(plt, bundled, "energy", "Free energy (arb.)", "Comparison: energy"),
    )
    cs(
        "fig_compare_absorbance",
        plot_overlay(
            plt,
            bundled,
            "absorbance",
            "Absorbance proxy",
            "Comparison: absorbance",
        ),
    )
    cs(
        "fig_compare_ionic",
        plot_overlay(
            plt,
            bundled,
            "ionic_strength_M",
            r"$I$ (mol L$^{-1}$)",
            "Comparison: ionic strength",
        ),
    )
    cs(
        "fig_compare_conductivity",
        plot_overlay(
            plt,
            bundled,
            "conductivity_proxy",
            r"$\kappa$ proxy",
            "Comparison: conductivity",
        ),
    )
    cs(
        "fig_compare_spectral",
        plot_overlay(
            plt,
            bundled,
            "dirac_gap",
            "Dirac gap",
            "Comparison: Dirac gap",
        ),
    )
    cs(
        "fig_compare_braid",
        plot_overlay(
            plt,
            bundled,
            "braid_reduced_length",
            "Reduced length",
            "Comparison: braid length",
        ),
    )

    # Bar chart final energy
    labels = list(summaries.keys())
    finals = [_f(summaries[k].get("final_energy")) for k in labels]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = range(len(labels))
    ax.bar(x, finals, color="#2563eb", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Final energy (arb.)")
    ax.set_title("Final energy by run")
    fig.tight_layout()
    cs("fig_compare_bars", fig)

    (met_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (met_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        keys = sorted({k for s in summaries.values() for k in s.keys()})
        w.writerow(["run"] + keys)
        for lab, s in summaries.items():
            w.writerow([lab] + [s.get(k, "") for k in keys])

    stems_cmp = [
        "fig_compare_energy",
        "fig_compare_absorbance",
        "fig_compare_ionic",
        "fig_compare_conductivity",
        "fig_compare_spectral",
        "fig_compare_braid",
        "fig_compare_bars",
    ]
    ftex = ["% Comparison figures", ""]
    for stem in stems_cmp:
        cap = tex_escape(f"Comparison: {stem}")
        gname = tex_escape_graphics_filename(f"{stem}.pdf")
        lbl = latex_safe_label(f"fig:{stem}")
        ftex.extend(
            [
                "\\begin{figure}[t]",
                "  \\centering",
                f"  \\includegraphics[width=0.9\\linewidth]{{{gname}}}",
                f"  \\caption{{{cap}}}",
                f"  \\label{{{lbl}}}",
                "\\end{figure}",
                "",
            ]
        )
    (tex_dir / "figures.tex").write_text("\n".join(ftex), encoding="utf-8")
    (tex_dir / "captions.tex").write_text("% See figures.tex captions\n", encoding="utf-8")
    tlines = [
        "% Auto-generated comparison table",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Final energy and ionic strength by run.}",
        "  \\label{tab:srg_compare}",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        "    Run & Final energy & $I$ final / mol\\,L$^{-1}$ \\\\",
        "    \\midrule",
    ]
    for lab in sorted(summaries.keys()):
        s = summaries[lab]
        lab_tex = tex_escape(lab)
        tlines.append(
            f"    {lab_tex} & {latex_table_cell(s.get('final_energy'))} & {latex_table_cell(s.get('final_ionic_strength'))} \\\\"
        )
    tlines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    (tex_dir / "table_metrics.tex").write_text("\n".join(tlines), encoding="utf-8")
    esc_labels = ", ".join(tex_escape(x) for x in labels)
    (tex_dir / "results_summary.tex").write_text(
        "% Multi-run comparison — see metrics/summary_metrics.json\n"
        "\\paragraph{Comparison summary.}\n"
        f"Compared {len(labels)} runs: {esc_labels}.\n",
        encoding="utf-8",
    )

    settings_lines = ["Comparison runs:"] + [tex_escape(str(p)) for p in run_dirs]
    write_mini_report(
        rep_dir / "mini_report.tex",
        title="SRG multi-run comparison",
        run_label="comparison",
        settings_lines=settings_lines,
        summary_tex_rel="../tex/results_summary",
        table_tex_rel="../tex/table_metrics",
        figure_stems=[f"{s}.pdf" for s in stems_cmp],
    )

    _flush_warnings()
    return {"paper_root": str(paper_root), "summaries": summaries}


# -----------------------------------------------------------------------------
# Batch
# -----------------------------------------------------------------------------


def process_batch(runs_root: Path, paper_root: Path, smooth_window: int) -> None:
    runs_root = runs_root.resolve()
    paper_root = paper_root.resolve()
    discovered = discover_runs(runs_root)
    if not discovered:
        print(f"No runs under {runs_root}")
        return
    for rd in discovered:
        out = paper_root / rd.name
        print(f"Processing {rd.name} -> {out}")
        process_single(rd, out, smooth_window=smooth_window)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _LOG
    _LOG = []
    ap = argparse.ArgumentParser(description="SRG paper figures + LaTeX pipeline")
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--paper_dir", type=str, required=True)
    ap.add_argument("--compare", type=str, nargs="+", default=None)
    ap.add_argument("--all", type=str, dest="batch_root", default=None, metavar="RUNS_ROOT")
    ap.add_argument("--smooth_window", type=int, default=1)
    args = ap.parse_args(list(argv) if argv is not None else None)

    paper = Path(args.paper_dir)

    if args.compare:
        process_compare([Path(p) for p in args.compare], paper, smooth_window=args.smooth_window)
        return 0

    if args.batch_root:
        process_batch(Path(args.batch_root), paper, args.smooth_window)
        return 0

    if args.run_dir:
        process_single(Path(args.run_dir), paper, smooth_window=args.smooth_window)
        return 0

    ap.error("Provide --run_dir, --compare, or --all")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
