#!/usr/bin/env bash
# SRG Chemical Computing Platform v6.1 — Example: temperature sweep (thermodynamic knob)
#
# Demonstrates:
#   - Repeating the same composition at multiple temperature_K (dielectric uses water-like default)
#   - Electrolyte A/B coefficients and Debye length depend on T through electrolyte_metrics()
#   - Separate run directories for reproducible comparison of metrics_bundle.jsonl / figures
#
# Each sub-run uses fewer steps so the sweep finishes quickly; increase STEPS for publication-style runs.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_OUT="runs/example_temperature_sweep"
STEPS="${STEPS:-120}"

# Kelvin values (extend or edit as needed)
TEMPS=(278.15 298.15 318.15)

for T in "${TEMPS[@]}"; do
  TAG=$(printf "%.0f" "$(echo "$T" | awk '{print $1-273.15}')")C
  OUT="${BASE_OUT}/run_${TAG}_T${T}"
  mkdir -p "$OUT"
  echo "=== Running T = ${T} K (${TAG}) -> ${OUT} ==="
  python3 -m srg_hbond.visual_demo_chem \
    --preset nacl \
    --n_water 38 \
    --n_na 8 \
    --n_cl 8 \
    --electrolyte_model extended_dh \
    --temperature_K "$T" \
    --dielectric 78.54 \
    --box_volume_l 1e-22 \
    --steps "$STEPS" \
    --mode boltzmann \
    --temperature 0.5 \
    --seed 17 \
    --auto_png \
    --out_dir "$OUT"
done

echo "Sweep complete. Compare runs under: $BASE_OUT/"
echo "Each run has figures/electrolyte_activity.png and metrics_bundle.jsonl"
