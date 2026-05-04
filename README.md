# SRG Chemical Computing Platform v6.1

## 1. Project overview

**SRG Chemical Computing Platform** is a Python research framework that models aqueous solutions as **dynamic hydrogen-bond graphs** with **multi-species particles** (water, ions, polar and hydrophobic solutes, photochromic dyes). It combines **coarse-grained energy terms**, optional **electrolyte thermodynamics** (Debye–Hückel family, Davies, Pitzer-like proxy, Onsager conductivity), **toy photochemistry**, **spectral / NCG observables** on the graph, **braid statistics** on trajectories, and **optimization hooks** (RL / ant colony). Outputs include **step-wise JSON** (`metrics_bundle.jsonl`), **HTML replay**, optional **live dashboard**, and **PNG figure packs**.

This is a **research / MVP** stack for experimentation and visualization—not a validated atomistic force field or MD engine.

---

## 2. Features

- **Chemistry:** Multi-species nodes, weighted edges, H-bond / solvation / hydrophobic / Coulomb-style terms; optional kJ/mol-inspired scaling (`--use_physical_params`).
- **Electrolyte models:** `dh`, `extended_dh`, `davies`, `pitzer` (simplified 1:1 proxy); ionic strength, activity coefficients, Debye length; screening coupled to metrics where enabled.
- **Photochemistry:** Light intensity drives dye `A → B` transitions (toy rates); absorbance / multi-band proxies in output.
- **RL / ACO:** Separate training entry points for chemistry ACO, NCG/braid RL and ACO (`train_*` modules).
- **Spectral + braid metrics:** Graph Dirac / gaps / smoothness (NCG layer); braid word length, writhe, entropy from actions.
- **Visualization:** Static `visualizer.html`, optional live `dashboard.html` + `frames_live.json`, automatic PNG reports (`--auto_png`).

---

## 3. Installation

```bash
git clone <YOUR_REPO_URL> srg_hbond_chem_mvp_v6
cd srg_hbond_chem_mvp_v6
pip install -r requirements.txt
```

**Minimal alternative** (no requirements file):

```bash
pip install "numpy>=1.20" "matplotlib>=3.5"
```

Ensure the package root is on `PYTHONPATH` when running as a module from the repo root (see [Troubleshooting](#11-troubleshooting)):

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 4. Quick start (most important)

Run a short mixed-system simulation with **live JSON refresh**, then serve the run directory:

```bash
python3 -m srg_hbond.visual_demo_chem \
  --steps 100 \
  --preset mixed \
  --live \
  --flush_every 5 \
  --out_dir runs/quick
```

**Dashboard** (polling `frames_live.json` during the run):

```bash
python3 -m http.server 8000 --directory runs/quick
```

Open **http://127.0.0.1:8000/dashboard.html** in a browser.

When the run finishes, open **http://127.0.0.1:8000/visualizer.html** for full replay (or open `runs/quick/visualizer.html` directly from disk).

---

## 5. Core commands

### Basic run

```bash
python3 -m srg_hbond.visual_demo_chem --steps 200 --preset mixed --out_dir runs/basic
```

### Custom chemistry

```bash
python3 -m srg_hbond.visual_demo_chem \
  --steps 250 \
  --n_water 40 --n_na 4 --n_cl 4 \
  --n_polar 3 --n_hydrophobic 4 \
  --n_dye_a 3 --n_dye_b 0 \
  --light 0.65 \
  --out_dir runs/custom_chem
```

### Electrolyte models

```bash
python3 -m srg_hbond.visual_demo_chem \
  --steps 220 \
  --preset nacl --n_na 6 --n_cl 6 \
  --electrolyte_model davies \
  --use_onsager \
  --temperature_K 298.15 \
  --box_volume_l 1e-22 \
  --auto_png \
  --out_dir runs/electrolyte_davies
```

```bash
python3 -m srg_hbond.visual_demo_chem \
  --steps 220 \
  --preset nacl --n_na 18 --n_cl 18 \
  --electrolyte_model pitzer \
  --box_volume_l 8e-23 \
  --auto_png \
  --out_dir runs/electrolyte_pitzer
```

### Photochemistry

```bash
python3 -m srg_hbond.visual_demo_chem \
  --steps 200 \
  --preset photo \
  --light 0.85 \
  --auto_png \
  --out_dir runs/photo
```

### PNG generation

```bash
python3 -m srg_hbond.visual_demo_chem --steps 150 --preset mixed --auto_png --out_dir runs/with_png
```

Regenerate figures from an existing run (if supported by `plot_chem`):

```bash
python3 -m srg_hbond.plot_chem --run_dir runs/with_png
```

### RL / ACO (available entry points)

Ant colony on chemistry environment:

```bash
python3 -m srg_hbond.train_aco_chem --preset photo --steps 150 --ants 16 --out_dir runs/aco_chem
```

RL / ACO on NCG or braid-focused envs:

```bash
python3 -m srg_hbond.train_rl_ncg --out_dir runs/rl_ncg
python3 -m srg_hbond.train_aco_ncg --out_dir runs/aco_ncg
python3 -m srg_hbond.train_rl_braid --out_dir runs/rl_braid
python3 -m srg_hbond.train_aco_braid --out_dir runs/aco_braid
```

Use `--help` on each module for full flags.

---

## 6. Examples

Bundled shell drivers under `examples/` (executable, self-contained):

```bash
chmod +x examples/*.sh
./examples/water_nacl_davies.sh
./examples/water_dye_photo.sh
./examples/high_salt_pitzer.sh
./examples/hydrophobic_cluster.sh
./examples/temperature_sweep.sh
./examples/full_system.sh
```

See **`examples/README.md`** for intent, expected outputs, and physics notes.

---

## 7. Output structure

Each `--out_dir` run typically contains:

| Path | Description |
|------|-------------|
| `metrics_bundle.jsonl` | One JSON record per step (preferred for streaming / replotting). |
| `frames.json` | Full trajectory array (pretty-printed). |
| `visualizer.html` | Static replay: graph + braid strip + metrics. |
| `frames_live.json` | Written when `--live` is set; overwritten periodically with all frames so far. |
| `dashboard.html` | Simple live view that polls `frames_live.json` (created with `--live`). |
| `figures/` | PNGs when `--auto_png` is used (`energy.png`, `energy_terms.png`, electrolyte plots, etc.). |
| `electrolyte_params_report.json` | Metadata when an electrolyte model or `--use_onsager` is active. |
| `physical_params_report.json` | Written when `--use_physical_params` is set. |

Some older docs refer to `history.json`; this codebase primarily uses **`frames.json`** and **`metrics_bundle.jsonl`**.

---

## 8. Visualization

**Live dashboard:** run with `--live`, start `python3 -m http.server <port> --directory <out_dir>`, open `dashboard.html`.

**Full visualizer:** after the run completes, open `visualizer.html` from the same directory (via HTTP server or `file://`). The visualizer expects embedded frame data in `frames.json` / inlined in HTML.

---

## 9. Key parameters

| Flag | Role |
|------|------|
| `--n_water` | Number of water-like nodes. |
| `--n_na`, `--n_cl` | Sodium / chloride counts (electrolyte composition). |
| `--n_polar`, `--n_hydrophobic` | Polar / hydrophobic solute counts. |
| `--n_dye_a`, `--n_dye_b` | Photochromic educt / product counts. |
| `--light` | Light intensity `0…1` for photochemistry. |
| `--temperature` | Boltzmann MC temperature for **action selection** in `visual_demo_chem` (not Kelvin). |
| `--temperature_K` | Physical temperature (K) for electrolyte / optional physical energy layer. |
| `--electrolyte_model` | `none`, `dh`, `extended_dh`, `davies`, `pitzer`. |
| `--auto_png` | Write figure bundle under `figures/`. |
| `--live` | Emit `dashboard.html` and refresh `frames_live.json` during the run. |
| `--flush_every` | Steps between live JSON writes when `--live` is set. |
| `--use_physical_params` | Use coarse kJ/mol-style energy pathway where implemented. |
| `--use_onsager` | Add conductivity proxy to electrolyte metrics. |

---

## 10. Example experiments (conceptual)

- **Water + NaCl:** Davies or extended DH / Pitzer; inspect ionic strength, γ, λ_D, conductivity proxy PNGs.
- **Photochemical dye:** `--preset photo` or dye counts + `--light`; watch absorbance RGB / spectrum plots.
- **Hydrophobic clustering:** `--preset hydrophobic` or high `--n_hydrophobic`; watch hydrophobic energy term and graph clustering.
- **Full system:** ions + polar + hydrophobic + dye + light + electrolyte + physical params — see `examples/full_system.sh`.

---

## 11. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: srg_hbond` | Run from repository root: `export PYTHONPATH="$(pwd):${PYTHONPATH}"` or `pip install -e .` if you add a `setup.cfg` / `pyproject.toml`. |
| Dashboard frozen / empty | Use **`--live`** on `visual_demo_chem`; ensure you opened **`dashboard.html`** and the server root is the **same** `--out_dir`. |
| `python` vs `python3` | On macOS/Linux use **`python3`** consistently. |
| Missing `figures/` | Pass **`--auto_png`** or run `plot_chem` / `figures.generate_png_report` on a directory that contains `metrics_bundle.jsonl` or `frames.json`. |
| `history.json` not found | This project uses **`frames.json`** + **`metrics_bundle.jsonl`**; see §7. |

---

## 12. Advanced usage

- **RL / ACO:** Use `train_rl_*` and `train_aco_*` modules for policy / pheromone optimization on NCG, braid, or chemistry-flavored envs; tune reward weights (`--lambda_ncg`, etc.) in env constructors / CLI where exposed. **`train_rl_*` requires PyTorch:** `pip install torch`.
- **Spectral analysis:** NCG metrics are logged per frame (`dirac_gap`, `laplacian_gap`, `ncg_smoothness`); correlate with energy decomposition in `metrics_bundle.jsonl`.
- **Braid metrics:** `braid_reduced_length`, `braid_writhe`, events arrays—compare with graph moves in `visualizer.html`.

---

## License / citation

Add your license and preferred citation if distributing publicly.

---

## Version note

This README targets **v6.1** feature set (electrolyte layer, `metrics_bundle.jsonl`, multi-band absorbance where enabled). For behavior details, see inline module docstrings and `examples/README.md`.
