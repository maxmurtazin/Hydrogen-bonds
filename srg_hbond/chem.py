from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

WATER='water'; SOLUTE='solute'; NA='Na+'; CL='Cl-'; POLAR='polar'; HYDRO='hydrophobic'; DYE_A='dye_A'; DYE_B='dye_B'

@dataclass(frozen=True)
class Species:
    name: str
    charge: float = 0.0
    hbond_donor: int = 0
    hbond_acceptor: int = 0
    polarity: float = 0.0
    hydrophobicity: float = 0.0
    radius: float = 1.0
    color: str = '#777777'
    absorbance: float = 0.0
    # Optional multi-wavelength absorption proxies: wavelength key (nm) -> strength in [0,1]
    absorbance_bands_nm: Tuple[Tuple[str, float], ...] = ()

SPECIES: Dict[str, Species] = {
    WATER: Species(WATER, 0, 2, 2, 1.0, 0.0, 1.0, '#4c8dff', 0.05, ()),
    SOLUTE: Species(SOLUTE, 0, 1, 2, 0.7, 0.1, 1.2, '#e74c3c', 0.25, ()),
    NA: Species(NA, +1, 0, 0, 1.0, 0.0, 0.8, '#ffcc33', 0.02, ()),
    CL: Species(CL, -1, 0, 0, 0.8, 0.0, 1.1, '#88dd88', 0.02, ()),
    POLAR: Species(POLAR, 0, 1, 2, 0.85, 0.0, 1.1, '#b36bff', 0.35, ()),
    HYDRO: Species(HYDRO, 0, 0, 0, 0.05, 1.0, 1.2, '#444444', 0.15, ()),
    DYE_A: Species(DYE_A, 0, 1, 1, 0.65, 0.2, 1.2, '#ff4fc3', 0.85, (("430", 0.88), ("530", 0.42), ("630", 0.10))),
    DYE_B: Species(DYE_B, 0, 0, 2, 0.55, 0.4, 1.2, '#00bcd4', 0.55, (("430", 0.18), ("530", 0.72), ("630", 0.48))),
}


def species_absorbance_bands(spec: Species) -> Dict[str, float]:
    if spec.absorbance_bands_nm:
        return {k: float(v) for k, v in spec.absorbance_bands_nm}
    return {"default": float(spec.absorbance)}

@dataclass
class Node:
    species: str
    x: float
    y: float
    state: str = 'ground'

    @property
    def spec(self): return SPECIES[self.species]


def pair_params(a: str, b: str) -> Dict[str,float]:
    sa, sb = SPECIES[a], SPECIES[b]
    # favorable if donor/acceptor + polarity; ion-water solvation; hydrophobe-hydrophobe clustering
    hbond_capacity = min(sa.hbond_donor + sa.hbond_acceptor, sb.hbond_donor + sb.hbond_acceptor)
    hbond = -0.22 * hbond_capacity * (sa.polarity + sb.polarity) / 2
    ion_solv = 0.0
    if (abs(sa.charge) > 0 and b == WATER) or (abs(sb.charge) > 0 and a == WATER): ion_solv = -1.6
    if abs(sa.charge) > 0 and abs(sb.charge) > 0 and sa.charge * sb.charge < 0: ion_solv += -0.9
    hydro = 0.0
    if sa.hydrophobicity > 0.6 and sb.hydrophobicity > 0.6: hydro = -0.75
    elif sa.hydrophobicity > 0.6 or sb.hydrophobicity > 0.6: hydro = +0.35
    return {'hbond':hbond, 'ion_solv':ion_solv, 'hydrophobic':hydro, 'qq':sa.charge*sb.charge}

REACTIONS = {
    'photo_dye_A_to_B': {'reactant': DYE_A, 'product': DYE_B, 'requires_light': True},
    'thermal_B_to_A': {'reactant': DYE_B, 'product': DYE_A, 'requires_light': False},
}
