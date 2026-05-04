#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: dilute NaCl + Davies activity model
#
# Demonstrates:
#   - Aqueous NaCl at modest concentration (activity coefficients γ± < 1)
#   - Davies equation for log γ (semi-empirical, typical use ~I ≲ 0.5 M qualitative)
#   - Ionic strength I, Debye length λ_D, and screening tied to electrolyte_metrics()
#   - Optional Onsager/Kohlrausch molar conductivity proxy (--use_onsager)
#
# Outputs under runs/example_water_nacl_davies/:
#   metrics_bundle.jsonl, frames.json, visualizer.html, figures/*.png (with --auto_png)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="runs/example_water_nacl_davies"
mkdir -p "$OUT"

python3 -m srg_hbond.visual_demo_chem \
  --preset nacl \
  --n_water 42 \
  --n_na 6 \
  --n_cl 6 \
  --electrolyte_model davies \
  --temperature_K 298.15 \
  --dielectric 78.54 \
  --box_volume_l 1e-22 \
  --use_onsager \
  --steps 220 \
  --mode boltzmann \
  --temperature 0.52 \
  --lambda_ncg 0.025 \
  --lambda_gap 0.008 \
  --lambda_braid 0.004 \
  --seed 11 \
  --auto_png \
  --out_dir "$OUT"

echo "Done. Open: $OUT/visualizer.html"
echo "Figures: $OUT/figures/"
