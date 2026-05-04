from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import math

from .chem import SPECIES

# Coarse electrolyte layer for SRG-Chem.
# Models are intended for qualitative simulation/visualization, not prediction-grade electrolyte thermodynamics.

NA = 6.02214076e23
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
R_KJ_MOL_K = 0.00831446261815324
LN10 = math.log(10.0)

@dataclass
class ElectrolyteSettings:
    model: str = "none"  # none, dh, extended_dh, davies, pitzer
    use_onsager: bool = False
    box_volume_l: float = 1e-22
    temperature_K: float = 298.15
    dielectric: float = 78.54
    ion_size_nm: float = 0.35
    # Pitzer-like parameters for a 1:1 electrolyte, roughly NaCl-like defaults.
    pitzer_beta0: float = 0.0765
    pitzer_beta1: float = 0.2664
    pitzer_cphi: float = 0.00127
    pitzer_alpha: float = 2.0
    pitzer_b: float = 1.2
    # Onsager/Kohlrausch proxy constants.
    lambda0_s_cm2_mol: float = 126.45  # NaCl limiting molar conductivity around 25 C
    onsager_K: float = 60.0            # empirical sqrt(c) slope proxy


def _counts_to_molarity(n: int, box_volume_l: float) -> float:
    if box_volume_l <= 0:
        return 0.0
    return max(0.0, n / (NA * box_volume_l))


def electrolyte_concentrations(species_counts: Dict[str, int], box_volume_l: float) -> Dict[str, float]:
    """Molarity (mol/L) for every charged species present in the registry."""
    out: Dict[str, float] = {}
    for name, n in species_counts.items():
        sp = SPECIES.get(name)
        if sp is None or abs(sp.charge) < 1e-12:
            continue
        c = _counts_to_molarity(int(n), box_volume_l)
        if c > 0.0:
            out[name] = c
    return out


def ionic_strength_M(conc: Dict[str, float]) -> float:
    # I = 1/2 sum c_i z_i^2
    s = 0.0
    for name, c in conc.items():
        if c <= 0.0:
            continue
        sp = SPECIES.get(name)
        if sp is None or abs(sp.charge) < 1e-12:
            continue
        z = float(sp.charge)
        s += c * z * z
    return 0.5 * s


def debye_length_nm(I_M: float, temperature_K: float = 298.15, dielectric: float = 78.54) -> float:
    if I_M <= 0:
        return float("inf")
    # kappa^2 = 2 e^2 N_A 1000 I / (epsilon_r epsilon0 k_B T)
    kappa2 = (2.0 * E_CHARGE**2 * NA * 1000.0 * I_M) / (max(1e-9, dielectric) * EPS0 * KB * temperature_K)
    if kappa2 <= 0:
        return float("inf")
    return 1e9 / math.sqrt(kappa2)


def debye_huckel_A(temperature_K: float = 298.15, dielectric: float = 78.54) -> float:
    # Approximate A for log10 gamma in water-like solvents. Gives ~0.509 at 298 K, eps~78.5.
    return 1.82483e6 / ((dielectric * temperature_K) ** 1.5)


def debye_huckel_B_nm(temperature_K: float = 298.15, dielectric: float = 78.54) -> float:
    # Approximate B in nm^-1 M^-1/2. Gives ~3.29 nm^-1 at 298 K in water.
    return 50.2916 / math.sqrt(dielectric * temperature_K)


def log10_gamma_ion(z: int, I_M: float, model: str, settings: ElectrolyteSettings) -> float:
    if model in ("none", None) or I_M <= 0:
        return 0.0
    sqrtI = math.sqrt(max(I_M, 0.0))
    A = debye_huckel_A(settings.temperature_K, settings.dielectric)
    B = debye_huckel_B_nm(settings.temperature_K, settings.dielectric)
    z2 = z * z
    if model == "dh":
        return -A * z2 * sqrtI
    if model == "extended_dh":
        return -(A * z2 * sqrtI) / (1.0 + B * settings.ion_size_nm * sqrtI)
    if model == "davies":
        return -A * z2 * (sqrtI / (1.0 + sqrtI) - 0.3 * I_M)
    if model == "pitzer":
        # Pitzer-like 1:1 mean activity coefficient proxy.
        # This is not a full Pitzer implementation; it gives a tunable non-DH curvature at finite molality.
        m = I_M
        f = -A * sqrtI / (1.0 + settings.pitzer_b * sqrtI)
        Bmx = settings.pitzer_beta0 + settings.pitzer_beta1 * math.exp(-settings.pitzer_alpha * sqrtI)
        ln_gamma_pm = f * LN10 + 2.0 * m * Bmx + 1.5 * (m**2) * settings.pitzer_cphi
        return ln_gamma_pm / LN10
    return 0.0


def onsager_conductivity_proxy(conc: Dict[str, float], settings: ElectrolyteSettings) -> Dict[str, float]:
    # Kohlrausch/Onsager-style: Lambda_m = Lambda0 - K sqrt(c).
    if not conc:
        c = 0.0
    else:
        na, cl = conc.get("Na+", 0.0), conc.get("Cl-", 0.0)
        if na > 0.0 and cl > 0.0:
            c = min(na, cl)
        else:
            I = ionic_strength_M(conc)
            c = max(I, 0.0)
    sqrtc = math.sqrt(max(c, 0.0))
    lambda_m = max(0.0, settings.lambda0_s_cm2_mol - settings.onsager_K * sqrtc)
    # specific conductivity proxy kappa ~ Lambda_m * c / 1000 (S/cm if Lambda in S cm^2 mol^-1 and c in mol/L)
    kappa_s_cm = lambda_m * c / 1000.0
    return {
        "molar_conductivity_s_cm2_mol": lambda_m,
        "conductivity_s_cm_proxy": kappa_s_cm,
        "conductivity_relative": kappa_s_cm / max(1e-12, settings.lambda0_s_cm2_mol * max(c,1e-12) / 1000.0),
    }


def electrolyte_metrics(species_counts: Dict[str, int], settings: ElectrolyteSettings) -> Dict[str, Any]:
    conc = electrolyte_concentrations(species_counts, settings.box_volume_l)
    I = ionic_strength_M(conc)
    lam = debye_length_nm(I, settings.temperature_K, settings.dielectric)
    logg_by: Dict[str, float] = {}
    g_by: Dict[str, float] = {}
    for name, c in conc.items():
        if c <= 0.0:
            continue
        sp = SPECIES.get(name)
        if sp is None or abs(sp.charge) < 1e-12:
            continue
        z_i = int(round(float(sp.charge)))
        lg = log10_gamma_ion(z_i, I, settings.model, settings)
        logg_by[name] = lg
        g_by[name] = 10.0**lg
    w_num = 0.0
    w_den = 0.0
    for name, c in conc.items():
        sp = SPECIES.get(name)
        if sp is None or abs(sp.charge) < 1e-12 or c <= 0.0:
            continue
        w = c * abs(float(sp.charge))
        w_num += w * logg_by.get(name, 0.0)
        w_den += w
    log_gamma_mean = w_num / w_den if w_den > 0.0 else 0.0
    gamma_mean = 10.0**log_gamma_mean
    out: Dict[str, Any] = {
        "model": settings.model,
        "concentration_M": conc,
        "ionic_strength_M": I,
        "debye_length_nm": lam,
        "A_DH": debye_huckel_A(settings.temperature_K, settings.dielectric),
        "B_DH_nm_inv": debye_huckel_B_nm(settings.temperature_K, settings.dielectric),
        "log10_gamma_by_species": logg_by,
        "gamma_by_species": g_by,
        "log10_gamma_mean": log_gamma_mean,
        "gamma_mean": gamma_mean,
    }
    if "Na+" in g_by:
        out["log10_gamma_na"] = logg_by["Na+"]
        out["gamma_na"] = g_by["Na+"]
    if "Cl-" in g_by:
        out["log10_gamma_cl"] = logg_by["Cl-"]
        out["gamma_cl"] = g_by["Cl-"]
    if settings.use_onsager:
        out.update(onsager_conductivity_proxy(conc, settings))
    return out


def electrolyte_excess_free_energy_kj_mol(metrics: Dict[str, Any], temperature_K: float, n_ions: int) -> float:
    # Toy excess term: n_ions * RT ln(gamma_mean). Negative for gamma<1.
    gamma = max(1e-12, float(metrics.get("gamma_mean", 1.0)))
    return float(n_ions * R_KJ_MOL_K * temperature_K * math.log(gamma))


def screened_factor(r_nm: float, debye_nm: float) -> float:
    if not math.isfinite(debye_nm) or debye_nm <= 0:
        return 1.0
    return math.exp(-max(0.0, r_nm) / debye_nm)


def settings_report(settings: ElectrolyteSettings) -> Dict[str, Any]:
    return {
        "settings": asdict(settings),
        "warning": "Debye-Huckel/Davies are dilute-solution approximations; Pitzer here is a simplified 1:1 NaCl-like proxy, not a validated parameter database.",
        "validity_hint": "Use dh/extended_dh qualitatively at low ionic strength; Davies often semi-empirical up to moderate I; use full Pitzer parameters for quantitative concentrated-electrolyte work.",
    }
