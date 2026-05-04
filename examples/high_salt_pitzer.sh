#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: concentrated NaCl + Pitzer-like proxy
#
# Demonstrates:
#   - Elevated ionic strength I (smaller Debye length → stronger electrostatic screening)
#   - Pitzer-style mean activity coefficient proxy (curvature beyond Debye–Hückel)
#   - Coulomb / pair weights responding to λ_D via the electrolyte layer
#
# Note: The bundled Pitzer implementation is a simplified 1:1 NaCl-like proxy for
#       visualization and RL shaping — not a full Pitzer parameter database fit.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="runs/example_high_salt_pitzer"
mkdir -p "$OUT"

python3 -m srg_hbond.visual_demo_chem \
  --preset nacl \
  --n_water 36 \
  --n_na 22 \
  --n_cl 22 \
  --electrolyte_model pitzer \
  --pitzer_beta0 0.0765 \
  --pitzer_beta1 0.2664 \
  --pitzer_cphi 0.00127 \
  --temperature_K 298.15 \
  --dielectric 78.54 \
  --box_volume_l 8e-23 \
  --use_onsager \
  --steps 260 \
  --mode boltzmann \
  --temperature 0.48 \
  --lambda_ncg 0.02 \
  --lambda_braid 0.003 \
  --seed 23 \
  --auto_png \
  --out_dir "$OUT"

echo "Done. Open: $OUT/visualizer.html"
echo "Compare electrolyte_activity.png and debye_length.png under $OUT/figures/"
