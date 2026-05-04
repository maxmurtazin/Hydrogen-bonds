from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

# Coarse-grained physical parameter layer.
# Values are representative near 25 C and intentionally kept configurable.
# This MVP is NOT a quantitative molecular dynamics force field.

R_KJ_MOL_K = 0.00831446261815324
COULOMB_KJ_MOL_NM = 138.935456  # kJ mol^-1 nm e^-2

@dataclass(frozen=True)
class WaterPhysicalParams:
    temperature_K: float = 298.15
    molar_mass_g_mol: float = 18.01528
    density_g_cm3: float = 0.9970479        # water near 25 C
    dielectric: float = 78.54               # static dielectric near 25 C
    heat_capacity_J_mol_K: float = 75.3     # liquid water Cp near 25 C
    enthalpy_vap_kJ_mol: float = 44.0       # standard vap enthalpy around 25 C
    hbond_energy_kJ_mol: float = 20.0       # broad typical H-bond scale, not universal
    entropy_edge_J_mol_K: float = 1.0       # toy ordering penalty per graph edge

@dataclass(frozen=True)
class IonPhysicalParams:
    charge_e: float
    hydration_free_energy_kJ_mol: float
    preferred_coordination: float
    color: str

ION_PARAMS: Dict[str, IonPhysicalParams] = {
    # Single-ion hydration free energies are convention-dependent.
    # These representative values are used as tunable coarse-grained defaults.
    'Na+': IonPhysicalParams(+1.0, -365.0, 6.0, '#ffcc33'),
    'Cl-': IonPhysicalParams(-1.0, -340.0, 6.0, '#88dd88'),
}

@dataclass(frozen=True)
class CoarseChemParams:
    hydrophobe_water_penalty_kJ_mol: float = 5.0
    hydrophobe_cluster_bonus_kJ_mol: float = -6.0
    polar_hbond_scale_kJ_mol: float = 12.0
    ion_pair_bonus_kJ_mol: float = -12.0
    dye_photo_stabilization_kJ_mol: float = -5.0
    pH_edge_penalty_kJ_mol: float = 0.08


def thermal_kT_kJ_mol(T: float) -> float:
    return R_KJ_MOL_K * T


def default_parameter_report(temperature_K: float = 298.15, dielectric: float | None = None) -> Dict[str, Any]:
    water = WaterPhysicalParams(temperature_K=temperature_K, dielectric=dielectric or WaterPhysicalParams().dielectric)
    return {
        'water': asdict(water),
        'ions': {k: asdict(v) for k, v in ION_PARAMS.items()},
        'coarse': asdict(CoarseChemParams()),
        'thermal_kT_kJ_mol': thermal_kT_kJ_mol(temperature_K),
        'coulomb_kJ_mol_nm_e2': COULOMB_KJ_MOL_NM,
        'warning': 'Representative coarse-grained parameters; not a validated force field or MD model.'
    }
