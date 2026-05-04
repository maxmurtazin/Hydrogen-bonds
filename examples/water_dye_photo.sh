#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: photochromic dye + light intensity
#
# Demonstrates:
#   - Light-driven A→B isomerization (toy kinetics in ChemHBondEnv.step)
#   - Population-averaged absorbance proxy and multi-band spectrum (dye_A / dye_B bands)
#   - RGB colorimeter-style trace for dashboard / PNG reports
#
# Increase --steps or --light to push further toward the photostationary distribution.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="runs/example_water_dye_photo"
mkdir -p "$OUT"

python3 -m srg_hbond.visual_demo_chem \
  --preset photo \
  --n_water 34 \
  --n_polar 4 \
  --n_dye_a 8 \
  --n_dye_b 0 \
  --light 0.88 \
  --electrolyte_model none \
  --steps 280 \
  --mode boltzmann \
  --temperature 0.5 \
  --lambda_ncg 0.03 \
  --lambda_braid 0.006 \
  --seed 31 \
  --auto_png \
  --out_dir "$OUT"

echo "Done. Open: $OUT/visualizer.html"
echo "Inspect absorbance_spectrum.png and absorbance_rgb.png in $OUT/figures/"
