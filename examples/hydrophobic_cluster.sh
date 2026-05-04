#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: hydrophobic solute aggregation
#
# Demonstrates:
#   - Favorable hydrophobe–hydrophobe contacts vs hydrophobe–water penalties (graph energy)
#   - Boltzmann MC moves that explore clustered vs dispersed configurations
#   - Braid / NCG metrics as the graph rearranges (secondary readout of structural change)
#
# Expect: lower toy free energy when hydrophobic nodes share edges (check energy_terms.png).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="runs/example_hydrophobic_cluster"
mkdir -p "$OUT"

python3 -m srg_hbond.visual_demo_chem \
  --preset hydrophobic \
  --n_water 26 \
  --n_hydrophobic 14 \
  --n_polar 0 \
  --n_na 0 \
  --n_cl 0 \
  --electrolyte_model none \
  --steps 300 \
  --mode boltzmann \
  --temperature 0.45 \
  --boltzmann_candidates 64 \
  --lambda_ncg 0.035 \
  --lambda_gap 0.012 \
  --lambda_braid 0.007 \
  --seed 41 \
  --auto_png \
  --out_dir "$OUT"

echo "Done. Open: $OUT/visualizer.html"
echo "Watch hydrophobic term in $OUT/figures/energy_terms.png"
