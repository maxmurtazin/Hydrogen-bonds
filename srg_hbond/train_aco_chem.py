from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from .env_chem import ChemHBondEnv, ChemConfig
from .ncg import ncg_reward_correction
from .braid import BraidTracker, braid_reward_correction
from .figures import generate_png_report
from .progress import ETA, print_progress
from .physchem import default_parameter_report
from .electrolyte import ElectrolyteSettings, settings_report


def _make_cfg(args):
    presets = {
        "nacl": dict(n_na=4, n_cl=4, n_polar=0, n_hydrophobic=0, n_dye_a=0, n_dye_b=0),
        "polar": dict(n_na=0, n_cl=0, n_polar=8, n_hydrophobic=0, n_dye_a=0, n_dye_b=0),
        "hydrophobic": dict(n_na=0, n_cl=0, n_polar=0, n_hydrophobic=10, n_dye_a=0, n_dye_b=0),
        "dye": dict(n_na=0, n_cl=0, n_polar=2, n_hydrophobic=0, n_dye_a=6, n_dye_b=0),
        "photo": dict(n_na=0, n_cl=0, n_polar=2, n_hydrophobic=0, n_dye_a=6, n_dye_b=0),
        "mixed": dict(n_na=3, n_cl=3, n_polar=4, n_hydrophobic=5, n_dye_a=3, n_dye_b=0),
    }
    counts = dict(presets[args.preset])
    for key in ["n_na", "n_cl", "n_polar", "n_hydrophobic", "n_dye_a", "n_dye_b"]:
        val = getattr(args, key, None)
        if val is not None:
            counts[key] = val
    light = args.light if args.light is not None else (0.9 if args.preset == "photo" else 0.0)
    return ChemConfig(
        n_water=args.n_water, seed=args.seed, light_intensity=light,
        pH=args.pH, ionic_strength=args.ionic_strength,
        use_physical_params=args.use_physical_params,
        temperature_K=args.temperature_K,
        dielectric=args.dielectric,
        energy_scale=args.energy_scale,
        electrolyte_model=args.electrolyte_model,
        use_onsager=args.use_onsager,
        box_volume_l=args.box_volume_l,
        ion_size_nm=args.ion_size_nm,
        pitzer_beta0=args.pitzer_beta0,
        pitzer_beta1=args.pitzer_beta1,
        pitzer_cphi=args.pitzer_cphi,
        **counts
    )


def score_action(env, action, temperature=0.6):
    e0 = env.free_energy(); c = env.clone(); _, r, _ = c.step(action)
    de = c.free_energy() - e0
    return math.exp(-de / max(temperature, 1e-6)) * math.exp(max(-8.0, min(8.0, r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--ants', type=int, default=24)
    ap.add_argument('--n_water', type=int, default=28)
    ap.add_argument('--n_na', type=int, default=None, help='override Na+ count')
    ap.add_argument('--n_cl', type=int, default=None, help='override Cl- count')
    ap.add_argument('--n_polar', type=int, default=None, help='override polar solute count')
    ap.add_argument('--n_hydrophobic', type=int, default=None, help='override hydrophobic solute count')
    ap.add_argument('--n_dye_a', type=int, default=None, help='override photochromic dye_A count')
    ap.add_argument('--n_dye_b', type=int, default=None, help='override product dye_B count')
    ap.add_argument('--light', type=float, default=None, help='light intensity 0..1; overrides preset')
    ap.add_argument('--pH', type=float, default=7.0, help='toy pH proxy; affects H-bond compatibility')
    ap.add_argument('--ionic_strength', type=float, default=0.0, help='toy ionic strength proxy; screens Coulomb')
    ap.add_argument('--use_physical_params', action='store_true', help='use coarse-grained kJ/mol physical parameter layer')
    ap.add_argument('--temperature_K', type=float, default=298.15, help='physical temperature for kJ/mol layer')
    ap.add_argument('--dielectric', type=float, default=78.54, help='relative dielectric permittivity for screened Coulomb, water near 25C ~78.54')
    ap.add_argument('--energy_scale', type=float, default=0.02, help='scale kJ/mol-like physical energy into stable toy reward units')
    ap.add_argument('--electrolyte_model', choices=['none','dh','extended_dh','davies','pitzer'], default='none', help='activity/screening model for electrolyte layer')
    ap.add_argument('--use_debye_huckel', action='store_true', help='shortcut for --electrolyte_model dh')
    ap.add_argument('--use_onsager', action='store_true', help='enable Onsager/Kohlrausch conductivity proxy')
    ap.add_argument('--box_volume_l', type=float, default=1e-22, help='effective simulation volume in liters for concentration estimates')
    ap.add_argument('--ion_size_nm', type=float, default=0.35, help='ion-size parameter for extended Debye-Huckel')
    ap.add_argument('--pitzer_beta0', type=float, default=0.0765, help='Pitzer-like beta0 for 1:1 electrolyte proxy')
    ap.add_argument('--pitzer_beta1', type=float, default=0.2664, help='Pitzer-like beta1 for 1:1 electrolyte proxy')
    ap.add_argument('--pitzer_cphi', type=float, default=0.00127, help='Pitzer-like Cphi for 1:1 electrolyte proxy')
    ap.add_argument('--auto_png', action='store_true', help='generate PNG figures automatically after run')
    ap.add_argument('--preset', choices=['nacl','polar','hydrophobic','dye','photo','mixed'], default='mixed')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--lambda_ncg', type=float, default=0.03)
    ap.add_argument('--lambda_gap', type=float, default=0.01)
    ap.add_argument('--lambda_braid', type=float, default=0.006)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='runs/aco_chem_demo')
    ap.add_argument('--progress_every', type=int, default=10, help='print CLI progress every N steps')
    args = ap.parse_args(); random.seed(args.seed)
    if args.use_debye_huckel and args.electrolyte_model == 'none':
        args.electrolyte_model = 'dh' 
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.use_physical_params:
        (out/'physical_params_report.json').write_text(json.dumps(default_parameter_report(args.temperature_K, args.dielectric), indent=2))
    if args.electrolyte_model != 'none' or args.use_onsager:
        es = ElectrolyteSettings(model=args.electrolyte_model, use_onsager=args.use_onsager, box_volume_l=args.box_volume_l, temperature_K=args.temperature_K, dielectric=args.dielectric, ion_size_nm=args.ion_size_nm, pitzer_beta0=args.pitzer_beta0, pitzer_beta1=args.pitzer_beta1, pitzer_cphi=args.pitzer_cphi)
        (out/'electrolyte_params_report.json').write_text(json.dumps(settings_report(es), indent=2))
    env = ChemHBondEnv(_make_cfg(args))
    braid = BraidTracker(len(env.nodes))
    hist = []
    best = {'reward': -1e9, 'energy': env.free_energy(), 'step': 0}
    eta_tracker = ETA(args.steps)
    for step in range(args.steps):
        acts = env.possible_actions(limit_pairs=320)
        # ant proposals sampled by local Boltzmann desirability
        candidates = random.choices(acts, weights=[score_action(env,a,args.temperature) for a in acts], k=max(1,args.ants))
        scored = []
        for a in candidates:
            c = env.clone(); st, base_r, _ = c.step(a)
            ncg_corr, ncg = ncg_reward_correction(st, args.lambda_ncg, args.lambda_gap)
            # approximate braid regularizer if this action is accepted
            bt = BraidTracker(len(c.nodes)); bt.events = list(braid.events); bt.permutation = list(braid.permutation)
            bt.add_action(st['positions'], a, step=step)
            bm = bt.metrics(); b_corr = braid_reward_correction(bm, args.lambda_braid, 0.001, 0.0)
            scored.append((base_r + ncg_corr + b_corr, a, st, ncg, bm))
        total_r, action, st, ncg, bm = max(scored, key=lambda x: x[0])
        st, base_r, info = env.step(action)
        braid.add_action(st['positions'], action, step=step)
        ncg_corr, ncg = ncg_reward_correction(st, args.lambda_ncg, args.lambda_gap)
        bm = braid.metrics(); b_corr = braid_reward_correction(bm, args.lambda_braid, 0.001, 0.0)
        total_r = base_r + ncg_corr + b_corr
        if total_r > best['reward']:
            best = {'reward': float(total_r), 'energy': float(st['energy']), 'step': step, 'action': action}
        pm = eta_tracker.tick(step)
        rec = {'step': step, 'action': action, 'reward': float(total_r), 'base_reward': float(base_r), 'energy': float(st['energy']),
               'energy_terms': st.get('energy_terms',{}), 'electrolyte': st.get('electrolyte',{}), 'energy_terms_kj_mol': st.get('energy_terms_kj_mol',{}), 'energy_kj_mol': st.get('energy_kj_mol'), 'species_counts': st['species_counts'], 'absorbance_rgb': st.get('absorbance_rgb',[st['absorbance'], st['absorbance'], st['absorbance']]),
               'ncg_smoothness': ncg.ncg_smoothness, 'dirac_gap': ncg.spectral_gap, 'laplacian_gap': ncg.laplacian_gap,
               'braid_reduced_length': bm.reduced_length, 'braid_writhe': bm.writhe, 'n_edges': len(st['edges']), **pm}
        hist.append(rec)
        if args.progress_every and (step % args.progress_every == 0 or step + 1 == args.steps):
            print_progress(step, args.steps, pm, prefix='aco    ')
            print(f"    E={st['energy']:.3f} R={total_r:.3f} action={action} species={st['species_counts']}")
    (out/'history.json').write_text(json.dumps(hist, indent=2))
    (out/'best.json').write_text(json.dumps(best, indent=2))
    if args.auto_png:
        figs = generate_png_report(out, hist)
        print('wrote PNG figures:', out/'figures', f'({len(figs)} files)')
    print('wrote', out)

if __name__ == '__main__': main()
