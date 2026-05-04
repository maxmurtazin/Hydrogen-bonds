#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: full mixed system (all major features)
#
# Demonstrates:
#   - Ions + polar + hydrophobic + photochromic dye in one graph
#   - Davies electrolyte layer (I, γ, λ_D) + physical parameter scaling optional
#   - Photochemistry with light; NCG reward shaping; braid topology on trajectory
#   - Comprehensive PNG suite + metrics_bundle.jsonl for offline analysis
#
# This is the closest “integration test” style experiment for SRG-Chem v6.1 visual pipeline.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="runs/example_full_system"
mkdir -p "$OUT"

python3 -m srg_hbond.visual_demo_chem \
  --preset mixed \
  --n_water 40 \
  --n_na 5 \
  --n_cl 5 \
  --n_polar 5 \
  --n_hydrophobic 6 \
  --n_dye_a 4 \
  --n_dye_b 0 \
  --light 0.75 \
  --pH 7.2 \
  --electrolyte_model davies \
  --use_onsager \
  --use_physical_params \
  --temperature_K 298.15 \
  --dielectric 78.54 \
  --box_volume_l 1e-22 \
  --energy_scale 0.018 \
  --steps 320 \
  --mode boltzmann \
  --temperature 0.5 \
  --boltzmann_candidates 56 \
  --lambda_ncg 0.032 \
  --lambda_gap 0.014 \
  --lambda_braid 0.008 \
  --seed 101 \
  --auto_png \
  --out_dir "$OUT"

echo "Done. Open: $OUT/visualizer.html"
echo "Reports: $OUT/electrolyte_params_report.json $OUT/physical_params_report.json"
