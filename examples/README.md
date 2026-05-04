# SRG Chemical Computing Platform v6.1 — Example experiments

Runnable bash drivers for `python3 -m srg_hbond.visual_demo_chem`. Each script changes composition, electrolyte model, and thermodynamic/optimization knobs to highlight a different physical or chemical effect.

## Prerequisites

- Python 3 with project dependencies available (NumPy, Matplotlib for `--auto_png`).
- From the **repository root**:

```bash
chmod +x examples/*.sh
```

Run any example:

```bash
./examples/water_nacl_davies.sh
```

Scripts `cd` to the repo root automatically via `BASH_SOURCE`, so they work from any working directory.

## Output layout (every run)

| Artifact | Purpose |
|----------|---------|
| `metrics_bundle.jsonl` | One JSON object per step (streaming-friendly; preferred input for `figures.load_history`) |
| `frames.json` | Full run history (pretty-printed) |
| `visualizer.html` | Static replay: graph, braid panel, energy/metrics |
| `figures/*.png` | Auto-generated plots when using `--auto_png` |
| `electrolyte_params_report.json` | Written when an electrolyte model or `--use_onsager` is enabled |

## Viewing the dashboard / visualizer

### Static visualizer (recommended after a run)

Open `runs/<example>/visualizer.html` in a browser (double-click or “Open File”). No server required.

### Live dashboard (optional)

The stock `visualizer.html` is generated at the end of the run. For **live** updates during execution, add `--live` to the `python3 -m srg_hbond.visual_demo_chem` invocation inside a script; then:

```bash
cd runs/example_<name>
python3 -m http.server 8000
```

Open `http://localhost:8000/dashboard.html` for polling `frames_live.json`, or open `visualizer.html` after completion.

---

## Experiment catalog

### 1. `water_nacl_davies.sh` → `runs/example_water_nacl_davies`

**Intent:** Dilute-to-moderate **1:1 electrolyte** with the **Davies** activity model and **Onsager** conductivity proxy.

**Physics:** Ionic strength \(I\), mean activity trend (\(\gamma_\pm < 1\) at finite \(I\)), Debye screening length \(\lambda_D\), and Kohlrausch-style **molar conductivity** vs \(\sqrt{c}\) (qualitative).

**Expected behavior:** Smooth traces in `figures/electrolyte_activity.png` and `debye_length.png`; Coulomb-related energy terms respond to screening when the electrolyte layer is active.

---

### 2. `high_salt_pitzer.sh` → `runs/example_high_salt_pitzer`

**Intent:** **Higher salt loading** (more ions, slightly smaller effective volume) with the **Pitzer-like 1:1 proxy** — stronger non-DH curvature at finite concentration.

**Physics:** Short \(\lambda_D\), stronger pair-level screening; activity coefficients from the simplified Pitzer branch (not a full Pitzer database fit).

**Expected behavior:** Lower Debye length and more pronounced deviation of \(\gamma\) vs dilute DH — compare with the Davies example at similar visual settings.

---

### 3. `water_dye_photo.sh` → `runs/example_water_dye_photo`

**Intent:** **Photochemistry** — `dye_A` with high **light** intensity; tracks **absorbance** and **multi-wavelength** proxies.

**Physics:** Toy light-driven **A→B** moves in the environment; species carry band-resolved absorption proxies (`absorbance_spectrum` in state/metrics).

**Expected behavior:** Drift in `absorbance_rgb.png` / `absorbance_spectrum.png` as `dye_B` appears; energy terms may show photo stabilization contributions when physical scaling is enabled elsewhere.

---

### 4. `hydrophobic_cluster.sh` → `runs/example_hydrophobic_cluster`

**Intent:** **Hydrophobic aggregation** — many hydrophobic particles, no ions.

**Physics:** Favorable hydrophobe–hydrophobe edges vs penalties near water; Boltzmann exploration of clustered graphs.

**Expected behavior:** Bursts of favorable **hydrophobic** contribution in `energy_terms.png`; graph snapshots in `visualizer.html` show nonpolar nodes sharing edges.

---

### 5. `temperature_sweep.sh` → `runs/example_temperature_sweep/run_*`

**Intent:** **Thermal sweep** — same nominal composition at **278 / 298 / 318 K** subfolders.

**Physics:** Debye–Hückel **A(T)** and **B(T)** enter through `temperature_K`; optional comparison of \(\lambda_D\) and activity across folders.

**Expected behavior:** Three separate `figures/` trees; Debye length and activity coefficients shift with \(T\).

**Environment variable:** `STEPS=200 ./examples/temperature_sweep.sh` overrides per-run step count (default 120 for speed).

---

### 6. `full_system.sh` → `runs/example_full_system`

**Intent:** **Everything at once:** ions + polar + hydrophobic + **light** + dye + **Davies** + **Onsager** + **`--use_physical_params`** + strong NCG/braid shaping.

**Physics:** Integrated graph energy (HB, solvation, hydrophobic, Coulomb, entropy, photo, excess electrolyte term), spectral/NCG observables, braid complexity penalty.

**Expected behavior:** Richest `metrics_bundle.jsonl`; inspect `energy_terms.png`, `spectral_gaps.png`, `braid_length.png`, and electrolyte/conductivity panels together.

---

## Reproducibility notes

- Each script sets `--seed` explicitly.
- Coarse-grained energies are **qualitative**; interpret trends, not quantitative experimental prediction.
- For longer production runs, increase `--steps` inside the script or pass overrides by editing the script.

## Folder structure

```text
examples/
  README.md
  water_nacl_davies.sh
  high_salt_pitzer.sh
  water_dye_photo.sh
  hydrophobic_cluster.sh
  temperature_sweep.sh
  full_system.sh
runs/
  example_water_nacl_davies/
  example_high_salt_pitzer/
  example_water_dye_photo/
  example_hydrophobic_cluster/
  example_temperature_sweep/
    run_5C_T278.15/
    run_25C_T298.15/
    run_45C_T318.15/
  example_full_system/
```

(Run scripts once to materialize `runs/example_*`.)
