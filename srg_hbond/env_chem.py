from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any
import math, random, copy
import numpy as np
from .chem import *
from .physchem import WaterPhysicalParams, CoarseChemParams, ION_PARAMS, COULOMB_KJ_MOL_NM, thermal_kT_kJ_mol
from .electrolyte import ElectrolyteSettings, electrolyte_metrics, electrolyte_excess_free_energy_kj_mol, screened_factor

Action = Tuple[str, int, int]

@dataclass
class ChemConfig:
    n_water: int = 28
    n_na: int = 2
    n_cl: int = 2
    n_polar: int = 2
    n_hydrophobic: int = 3
    n_dye_a: int = 2
    n_dye_b: int = 0
    seed: int = 1
    box: float = 1.0
    edge_cutoff: float = 0.28
    k_dist: float = 1.8
    entropy_edge_penalty: float = 0.018
    coulomb_scale: float = 0.12
    light_intensity: float = 0.0
    pH: float = 7.0
    ionic_strength: float = 0.0
    use_physical_params: bool = False
    temperature_K: float = 298.15
    dielectric: float = 78.54
    energy_scale: float = 0.02  # maps kJ/mol-like physical terms into stable toy rewards
    electrolyte_model: str = "none"  # none, dh, extended_dh, davies, pitzer
    use_onsager: bool = False
    box_volume_l: float = 1e-22
    ion_size_nm: float = 0.35
    pitzer_beta0: float = 0.0765
    pitzer_beta1: float = 0.2664
    pitzer_cphi: float = 0.00127

class ChemHBondEnv:
    def __init__(self, cfg: ChemConfig):
        self.cfg=cfg; self.rng=random.Random(cfg.seed); self.nodes: List[Node]=[]; self.edges: Dict[Tuple[int,int],float]={}; self.t=0
        self._init_nodes(); self._init_edges()
    def _randpos(self): return self.rng.random()*self.cfg.box, self.rng.random()*self.cfg.box
    def _init_nodes(self):
        x,y=0.5,0.5; self.nodes.append(Node(SOLUTE,x,y))
        for sp,n in [(WATER,self.cfg.n_water),(NA,self.cfg.n_na),(CL,self.cfg.n_cl),(POLAR,self.cfg.n_polar),(HYDRO,self.cfg.n_hydrophobic),(DYE_A,self.cfg.n_dye_a),(DYE_B,self.cfg.n_dye_b)]:
            for _ in range(n): self.nodes.append(Node(sp,*self._randpos()))
    def _dist(self,i,j):
        a,b=self.nodes[i],self.nodes[j]; return math.hypot(a.x-b.x,a.y-b.y)+1e-6
    def _edge_key(self,i,j): return (i,j) if i<j else (j,i)

    def _coulomb_screening_scalar(self) -> float:
        """Pair-level Coulomb attenuation: tied to Debye length when electrolyte model is active."""
        if self.cfg.electrolyte_model != "none":
            lam = self.electrolyte_metrics().get("debye_length_nm", float("inf"))
            if lam is not None and math.isfinite(lam) and lam > 0.0:
                kappa_nm = 1.0 / max(lam, 1e-12)
                ref_nm = 1.0
                return 1.0 / (1.0 + kappa_nm * ref_nm)
            return 1.0
        return 1.0 / (1.0 + max(0.0, self.cfg.ionic_strength))

    def _compat_weight(self,i,j):
        d=self._dist(i,j); a,b=self.nodes[i].species,self.nodes[j].species; p=pair_params(a,b)
        screening = self._coulomb_screening_scalar()
        raw = -(p['hbond'] + p['ion_solv'] + min(0,p['hydrophobic']))
        raw += 0.15*abs(p['qq'])*screening
        # acidic/basic deviation perturbs H-bond compatibility (toy proxy)
        ph_factor = max(0.25, 1.0 - 0.035*abs(self.cfg.pH - 7.0))
        return max(0.02, raw * ph_factor) * math.exp(-self.cfg.k_dist*d)
    def _init_edges(self):
        n=len(self.nodes)
        for i in range(n):
            for j in range(i+1,n):
                if self._dist(i,j)<self.cfg.edge_cutoff and self.rng.random()<0.55:
                    self.edges[(i,j)] = self._compat_weight(i,j)
    def clone(self): return copy.deepcopy(self)
    def species_counts(self):
        out={k:0 for k in SPECIES}
        for node in self.nodes: out[node.species]=out.get(node.species,0)+1
        return out

    def electrolyte_settings(self):
        return ElectrolyteSettings(
            model=self.cfg.electrolyte_model,
            use_onsager=self.cfg.use_onsager,
            box_volume_l=self.cfg.box_volume_l,
            temperature_K=self.cfg.temperature_K,
            dielectric=self.cfg.dielectric,
            ion_size_nm=self.cfg.ion_size_nm,
            pitzer_beta0=self.cfg.pitzer_beta0,
            pitzer_beta1=self.cfg.pitzer_beta1,
            pitzer_cphi=self.cfg.pitzer_cphi,
        )

    def electrolyte_metrics(self):
        return electrolyte_metrics(self.species_counts(), self.electrolyte_settings())
    def energy_terms(self):
        if self.cfg.use_physical_params:
            return self.energy_terms_physical_scaled()
        terms={'hbond':0.0,'ion_solv':0.0,'hydrophobic':0.0,'coulomb':0.0,'hydrophobic_exposure':0.0,'entropy':0.0,'photo':0.0,'pH_penalty':0.0}
        em = self.electrolyte_metrics() if self.cfg.electrolyte_model != "none" else {}
        dh_nm = em.get('debye_length_nm', float('inf'))
        screening = self._coulomb_screening_scalar()
        for (i,j),w in self.edges.items():
            p=pair_params(self.nodes[i].species,self.nodes[j].species); d=self._dist(i,j)
            terms['hbond'] += p['hbond']*w
            terms['ion_solv'] += p['ion_solv']*w
            terms['hydrophobic'] += p['hydrophobic']*w
            terms['coulomb'] += self.cfg.coulomb_scale*p['qq']*screening*screened_factor(d, dh_nm)/d
        for i,node in enumerate(self.nodes):
            if node.species==HYDRO:
                deg=sum(1 for e in self.edges if i in e); terms['hydrophobic_exposure'] += 0.05*deg
        terms['entropy'] += self.cfg.entropy_edge_penalty * len(self.edges)
        if self.cfg.light_intensity>0:
            terms['photo'] += sum(-0.12*self.cfg.light_intensity for n in self.nodes if n.species==DYE_B)
        terms['pH_penalty'] += 0.002*abs(self.cfg.pH-7.0)*len(self.edges)
        if self.cfg.electrolyte_model != "none":
            terms['electrolyte_excess'] = 0.08 * math.log(max(1e-12, em.get('gamma_mean', 1.0))) * (self.species_counts().get(NA,0) + self.species_counts().get(CL,0))
        return {k:float(v) for k,v in terms.items()}

    def energy_terms_physical_kj_mol(self):
        """Representative coarse-grained energy terms in kJ/mol-like units.

        The terms are inspired by physical constants but remain graph-level
        approximations: no angle-resolved water model, no explicit force field,
        no validated hydration-shell statistics. Use for hypothesis generation.
        """
        water = WaterPhysicalParams(temperature_K=self.cfg.temperature_K, dielectric=self.cfg.dielectric)
        coarse = CoarseChemParams()
        terms={'hbond':0.0,'ion_solv':0.0,'hydrophobic':0.0,'coulomb':0.0,'hydrophobic_exposure':0.0,'entropy':0.0,'photo':0.0,'pH_penalty':0.0}
        em = self.electrolyte_metrics() if self.cfg.electrolyte_model != "none" else {}
        dh_nm = em.get('debye_length_nm', float('inf'))
        screening = self._coulomb_screening_scalar()
        # In this 2D graph, distances are box-fraction units. Map box length to ~1 nm.
        for (i,j),w in self.edges.items():
            sp_i, sp_j = self.nodes[i].species, self.nodes[j].species
            si, sj = SPECIES[sp_i], SPECIES[sp_j]
            d_nm = max(0.10, self._dist(i,j))
            # H-bond / polar contact contribution.
            if (si.hbond_donor + si.hbond_acceptor) and (sj.hbond_donor + sj.hbond_acceptor):
                donor_acceptor = min(si.hbond_donor + si.hbond_acceptor, sj.hbond_donor + sj.hbond_acceptor)
                polarity = (si.polarity + sj.polarity) / 2.0
                if sp_i == WATER and sp_j == WATER:
                    terms['hbond'] += -water.hbond_energy_kJ_mol * min(1.0, donor_acceptor/4.0) * polarity * w
                else:
                    terms['hbond'] += -coarse.polar_hbond_scale_kJ_mol * min(1.0, donor_acceptor/4.0) * polarity * w
            # Ion hydration: distribute single-ion hydration free energy over a preferred first shell.
            for ion, other in [(sp_i, sp_j), (sp_j, sp_i)]:
                if ion in ION_PARAMS and other == WATER:
                    ip = ION_PARAMS[ion]
                    terms['ion_solv'] += (ip.hydration_free_energy_kJ_mol / max(1.0, ip.preferred_coordination)) * w
            if sp_i in ION_PARAMS and sp_j in ION_PARAMS and si.charge * sj.charge < 0:
                terms['ion_solv'] += coarse.ion_pair_bonus_kJ_mol * w
            # Screened Coulomb in water.
            qq = si.charge * sj.charge
            if qq:
                terms['coulomb'] += (COULOMB_KJ_MOL_NM * qq / (max(1.0, self.cfg.dielectric) * d_nm)) * screening * screened_factor(d_nm, dh_nm)
            # Hydrophobic coarse-grained contact/exposure.
            if si.hydrophobicity > 0.6 and sj.hydrophobicity > 0.6:
                terms['hydrophobic'] += coarse.hydrophobe_cluster_bonus_kJ_mol * w
            elif si.hydrophobicity > 0.6 or sj.hydrophobicity > 0.6:
                terms['hydrophobic'] += coarse.hydrophobe_water_penalty_kJ_mol * w
        for i,node in enumerate(self.nodes):
            if node.species==HYDRO:
                deg=sum(1 for e in self.edges if i in e)
                terms['hydrophobic_exposure'] += 0.8*deg
        # Entropic ordering penalty: T * DeltaS for graph constraints.
        terms['entropy'] += (self.cfg.temperature_K * water.entropy_edge_J_mol_K / 1000.0) * len(self.edges)
        if self.cfg.light_intensity>0:
            terms['photo'] += sum(coarse.dye_photo_stabilization_kJ_mol*self.cfg.light_intensity for n in self.nodes if n.species==DYE_B)
        terms['pH_penalty'] += coarse.pH_edge_penalty_kJ_mol*abs(self.cfg.pH-7.0)*len(self.edges)
        if self.cfg.electrolyte_model != "none":
            n_ions = self.species_counts().get(NA,0) + self.species_counts().get(CL,0)
            terms['electrolyte_excess'] = electrolyte_excess_free_energy_kj_mol(em, self.cfg.temperature_K, n_ions)
        return {k:float(v) for k,v in terms.items()}

    def energy_terms_physical_scaled(self):
        raw = self.energy_terms_physical_kj_mol()
        return {k: float(v*self.cfg.energy_scale) for k,v in raw.items()}

    def free_energy(self): return float(sum(self.energy_terms().values()))
    def free_energy_kj_mol(self): return float(sum(self.energy_terms_physical_kj_mol().values()))
    def absorbance(self):
        return sum(n.spec.absorbance for n in self.nodes)/max(1,len(self.nodes))

    def absorbance_spectrum(self) -> Dict[str, float]:
        """Population-averaged absorption proxy per wavelength band (nm keys)."""
        acc: Dict[str, float] = {}
        for n in self.nodes:
            for k, v in species_absorbance_bands(n.spec).items():
                acc[k] = acc.get(k, 0.0) + float(v)
        nn = max(1, len(self.nodes))
        return {k: acc[k] / nn for k in sorted(acc.keys())}
    def absorbance_rgb(self):
        # toy 3-channel colorimeter proxy: channels respond differently to dye/polar/ions
        cnt=self.species_counts(); n=max(1,len(self.nodes))
        r=(0.04*cnt[WATER]+0.08*cnt[NA]+0.10*cnt[CL]+0.34*cnt[POLAR]+0.15*cnt[HYDRO]+0.92*cnt[DYE_A]+0.24*cnt[DYE_B])/n
        g=(0.05*cnt[WATER]+0.05*cnt[NA]+0.12*cnt[CL]+0.28*cnt[POLAR]+0.13*cnt[HYDRO]+0.24*cnt[DYE_A]+0.62*cnt[DYE_B])/n
        b=(0.07*cnt[WATER]+0.04*cnt[NA]+0.08*cnt[CL]+0.45*cnt[POLAR]+0.12*cnt[HYDRO]+0.76*cnt[DYE_A]+0.85*cnt[DYE_B])/n
        return [float(max(0,min(1,x))) for x in (r,g,b)]
    def possible_actions(self, limit_pairs=260):
        n=len(self.nodes); acts=[]; pairs=[]
        for _ in range(limit_pairs):
            i,j=self.rng.randrange(n), self.rng.randrange(n)
            if i==j: continue
            if i>j: i,j=j,i
            pairs.append((i,j))
        seen=set()
        for i,j in pairs:
            if (i,j) in seen: continue
            seen.add((i,j)); key=(i,j)
            if key in self.edges: acts.append(('break',i,j))
            else: acts.append(('form',i,j))
            acts.append(('swap',i,j)); acts.append(('move',i,0)); acts.append(('move',j,0))
            if self.nodes[i].species in (DYE_A,DYE_B) or self.nodes[j].species in (DYE_A,DYE_B): acts.append(('react',i,j))
        return acts or [('noop',0,0)]
    def step(self, action: Action):
        e0=self.free_energy(); kind,i,j=action; key=self._edge_key(i,j)
        if kind=='form' and i!=j:
            self.edges[key]=self._compat_weight(i,j)
        elif kind=='break': self.edges.pop(key,None)
        elif kind=='swap' and i!=j:
            self.nodes[i].x,self.nodes[j].x=self.nodes[j].x,self.nodes[i].x; self.nodes[i].y,self.nodes[j].y=self.nodes[j].y,self.nodes[i].y
        elif kind=='move':
            self.nodes[i].x=min(1,max(0,self.nodes[i].x+self.rng.uniform(-0.04,0.04))); self.nodes[i].y=min(1,max(0,self.nodes[i].y+self.rng.uniform(-0.04,0.04)))
        elif kind=='react': self._react(i,j)
        # weak spontaneous photochemistry if light is on
        if self.cfg.light_intensity > 0:
            for idx,node in enumerate(self.nodes):
                if node.species==DYE_A and self.rng.random() < 0.006*self.cfg.light_intensity:
                    node.species=DYE_B
        self.t += 1
        e1=self.free_energy(); return self.state(), -(e1-e0), {'delta_energy': e1-e0}
    def _react(self,i,j):
        # light-driven dye A -> B, dark thermal B -> A; local polar/water neighbor boosts reaction
        for idx in (i,j):
            if idx<0 or idx>=len(self.nodes): continue
            sp=self.nodes[idx].species
            neigh=[b if a==idx else a for a,b in self.edges if idx in (a,b)]
            boost=1+0.25*sum(self.nodes[k].species in (WATER,POLAR) for k in neigh)
            if sp==DYE_A and self.cfg.light_intensity*boost > self.rng.random(): self.nodes[idx].species=DYE_B
            elif sp==DYE_B and (0.04*boost) > self.rng.random(): self.nodes[idx].species=DYE_A
    def state(self):
        em = self.electrolyte_metrics() if self.cfg.electrolyte_model != "none" else {}
        return {
            'positions': np.array([[n.x,n.y] for n in self.nodes],dtype=np.float32),
            'species': [n.species for n in self.nodes],
            'edges': dict(self.edges),
            'energy': self.free_energy(),
            'energy_terms': self.energy_terms(),
            'energy_terms_kj_mol': self.energy_terms_physical_kj_mol() if self.cfg.use_physical_params else {},
            'energy_kj_mol': self.free_energy_kj_mol() if self.cfg.use_physical_params else None,
            'absorbance': self.absorbance(),
            'absorbance_spectrum': self.absorbance_spectrum(),
            'absorbance_rgb': self.absorbance_rgb(),
            'species_counts': self.species_counts(),
            'solvation_shell_degree': sum(1 for e in self.edges if 0 in e),
            'pH': self.cfg.pH,
            'ionic_strength': self.cfg.ionic_strength,
            'ionic_strength_live_M': float(em.get('ionic_strength_M', 0.0) or 0.0),
            'screening_scalar': self._coulomb_screening_scalar(),
            'light_intensity': self.cfg.light_intensity,
            'use_physical_params': self.cfg.use_physical_params,
            'temperature_K': self.cfg.temperature_K,
            'dielectric': self.cfg.dielectric,
            'energy_scale': self.cfg.energy_scale,
            'electrolyte': em,
        }
