from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from .env_chem import ChemHBondEnv, ChemConfig
from .chem import SPECIES
from .ncg import ncg_reward_correction
from .braid import BraidTracker, braid_reward_correction
from .figures import generate_png_report
from .metrics_bundle import MetricsJsonlWriter
from .progress import ETA, print_progress, fmt_seconds
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


def pick_boltzmann(env, temperature=0.55, limit_pairs=48):
    acts = env.possible_actions(limit_pairs=limit_pairs)
    e0 = env.free_energy(); weights=[]
    for a in acts:
        c = env.clone(); c.step(a); de = c.free_energy()-e0
        weights.append(math.exp(-de/max(temperature,1e-6)))
    s=sum(weights); r=random.random()*s; acc=0
    for a,w in zip(acts,weights):
        acc += w
        if acc >= r: return a
    return acts[-1]


def html(frames, palette):
    payload=json.dumps(frames); pal=json.dumps(palette)
    return f"""<!doctype html><html><head><meta charset='utf-8'/><title>SRG-Chem-NCG-Braid Visualizer</title>
<style>
body{{font-family:system-ui,Arial;margin:20px;background:#0f1117;color:#f4f4f5}} .wrap{{display:grid;grid-template-columns:680px 420px minmax(340px,1fr);gap:18px;align-items:start}} canvas{{background:#fff;border-radius:16px;box-shadow:0 8px 28px rgba(0,0,0,.25)}} .panel{{background:#181c27;border:1px solid #2a3042;padding:16px;border-radius:16px}} input{{width:100%}} .legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}} .pill{{display:flex;align-items:center;gap:6px;background:#242a3a;border-radius:999px;padding:4px 8px;font-size:13px}} .dot{{width:12px;height:12px;border-radius:50%;display:inline-block}} pre{{white-space:pre-wrap;line-height:1.35}} .bar{{height:10px;border-radius:8px;background:#333;overflow:hidden;margin:5px 0 12px}} .bar span{{display:block;height:100%}}
</style></head><body><h1>SRG-Chem: multi-species H-bond + NCG + braid</h1>
<div class='wrap'><canvas id='cv' width='680' height='680'></canvas><canvas id='bcv' width='420' height='680'></canvas><div class='panel'>
<button id='play'>Play</button><input id='slider' type='range' min='0' max='{len(frames)-1}' value='0'/>
<div class='bar'><span id='prog'></span></div><div id='legend' class='legend'></div><h3>Absorbance proxy RGB</h3><div class='bar'><span id='ar'></span></div><div class='bar'><span id='ag'></span></div><div class='bar'><span id='ab'></span></div><pre id='info'></pre>
</div></div><script>
const frames={payload}; const palette={pal}; const cv=document.getElementById('cv'), ctx=cv.getContext('2d'); const bcv=document.getElementById('bcv'), bctx=bcv.getContext('2d'); const slider=document.getElementById('slider'), info=document.getElementById('info'); let playing=false;
const legend=document.getElementById('legend'); Object.entries(palette).forEach(([k,c])=>{{const d=document.createElement('div'); d.className='pill'; d.innerHTML=`<span class="dot" style="background:${{c}}"></span>${{k}}`; legend.appendChild(d);}});
function drawBraid(k){{ const f=frames[k]; bctx.clearRect(0,0,420,680); bctx.fillStyle='#fff'; bctx.fillRect(0,0,420,680); const events=f.braid_events||[]; const maxGen=Math.max(1,Math.min(20,f.n_nodes-1)); bctx.strokeStyle='rgba(0,0,0,.12)'; for(let g=1;g<=maxGen;g++){{const x=30+(g-1)*360/Math.max(1,maxGen-1); bctx.beginPath(); bctx.moveTo(x,25); bctx.lineTo(x,650); bctx.stroke();}} for(let idx=0;idx<events.length;idx++){{const e=events[idx]; const x=30+(Math.min(e.gen,maxGen)-1)*360/Math.max(1,maxGen-1); const y=35+idx*600/Math.max(1,events.length-1); bctx.beginPath(); bctx.arc(x,y,4.5,0,Math.PI*2); bctx.fillStyle=e.dir>0?'#16a34a':'#f97316'; bctx.fill();}} bctx.fillStyle='#111'; bctx.font='13px system-ui'; bctx.fillText('Braid / exchange events',18,18); }}
function draw(k){{ const f=frames[k], P=f.positions; ctx.clearRect(0,0,680,680); ctx.fillStyle='#fff'; ctx.fillRect(0,0,680,680);
  for(const e of f.edges){{const a=P[e[0]], b=P[e[1]], w=e[2]; ctx.globalAlpha=Math.min(0.85,0.18+w); ctx.lineWidth=1+3*Math.min(1,w); ctx.strokeStyle=f.types[e[0]]==='water'&&f.types[e[1]]==='water'?'#8aa4d6':'#555'; ctx.beginPath(); ctx.moveTo(40+a[0]*600,40+a[1]*600); ctx.lineTo(40+b[0]*600,40+b[1]*600); ctx.stroke();}} ctx.globalAlpha=1;
  for(let i=0;i<P.length;i++){{const p=P[i], typ=f.types[i], col=palette[typ]||'#888'; const r=typ==='water'?6:10; ctx.beginPath(); ctx.arc(40+p[0]*600,40+p[1]*600,r,0,Math.PI*2); ctx.fillStyle=col; ctx.fill(); ctx.lineWidth=1.5; ctx.strokeStyle='#111'; ctx.stroke(); if(typ!=='water'){{ctx.fillStyle='#111'; ctx.font='10px system-ui'; ctx.fillText(i,44+p[0]*600,36+p[1]*600);}}}}
  const [r,g,b]=f.absorbance_rgb; document.getElementById('ar').style=`width:${{Math.min(100,r*100)}}%;background:#ef4444`; document.getElementById('ag').style=`width:${{Math.min(100,g*100)}}%;background:#22c55e`; document.getElementById('ab').style=`width:${{Math.min(100,b*100)}}%;background:#3b82f6`;
  info.textContent=`step: ${{f.step}}\naction: ${{f.action}}\nenergy: ${{f.energy.toFixed(4)}}\nreward: ${{f.reward.toFixed(4)}}\nterms: ${{JSON.stringify(f.energy_terms)}}\nspecies: ${{JSON.stringify(f.species_counts)}}\nabsorbance_rgb: ${{f.absorbance_rgb.map(x=>x.toFixed(3)).join(', ')}}\nNCG smoothness: ${{f.ncg_smoothness.toFixed(4)}}\nDirac gap: ${{f.dirac_gap.toFixed(4)}}\nLaplacian gap: ${{f.laplacian_gap.toFixed(4)}}\nbraid reduced length: ${{f.braid_reduced_length}}\nwrithe: ${{f.braid_writhe}}\nedges: ${{f.n_edges}}\nelectrolyte: ${{JSON.stringify(f.electrolyte||{{}})}}`;
  drawBraid(k); }}
slider.oninput=()=>draw(+slider.value); document.getElementById('play').onclick=()=>{{playing=!playing}}; setInterval(()=>{{if(playing){{slider.value=(+slider.value+1)%frames.length; draw(+slider.value);}}}},150); draw(0);
</script></body></html>"""



def live_dashboard_html():
    return """<!doctype html><html><head><meta charset='utf-8'/><title>SRG-Chem Live Dashboard</title>
<style>
body{font-family:system-ui,Arial;margin:20px;background:#0f1117;color:#f4f4f5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#181c27;border:1px solid #2a3042;border-radius:16px;padding:16px}.bar{height:16px;border-radius:999px;background:#303849;overflow:hidden}.bar span{display:block;height:100%;background:#38bdf8}canvas{background:#fff;border-radius:16px;width:100%;max-width:620px}pre{white-space:pre-wrap}.metric{font-size:18px;margin:8px 0}
</style></head><body><h1>SRG-Chem Live Dashboard</h1><div class='card'><div class='bar'><span id='prog'></span></div><div id='status' class='metric'>loading...</div></div><div class='grid'><canvas id='cv' width='620' height='620'></canvas><div class='card'><pre id='info'></pre></div></div>
<script>
let frames=[]; const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
function color(t){return {water:'#93c5fd','Na+':'#facc15','Cl-':'#22c55e',polar:'#c084fc',hydrophobic:'#fb7185',dye_A:'#f97316',dye_B:'#14b8a6'}[t]||'#888'}
function draw(f){ if(!f)return; const P=f.positions; ctx.clearRect(0,0,620,620); ctx.fillStyle='#fff'; ctx.fillRect(0,0,620,620); for(const e of f.edges){const a=P[e[0]],b=P[e[1]];ctx.globalAlpha=.45;ctx.lineWidth=1+2*e[2];ctx.strokeStyle='#334155';ctx.beginPath();ctx.moveTo(35+a[0]*550,35+a[1]*550);ctx.lineTo(35+b[0]*550,35+b[1]*550);ctx.stroke();} ctx.globalAlpha=1; for(let i=0;i<P.length;i++){const p=P[i],t=f.types[i];ctx.beginPath();ctx.arc(35+p[0]*550,35+p[1]*550,t==='water'?5:9,0,Math.PI*2);ctx.fillStyle=color(t);ctx.fill();ctx.strokeStyle='#111';ctx.stroke();} document.getElementById('prog').style.width=((f.progress||0)*100)+'%'; document.getElementById('status').innerText=`step ${f.step} | ${((f.progress||0)*100).toFixed(1)}% | ETA ${(f.eta_s||0).toFixed(1)}s | ${(f.steps_per_s||0).toFixed(2)} step/s`; document.getElementById('info').textContent=JSON.stringify({energy:f.energy,reward:f.reward,species:f.species_counts,terms:f.energy_terms,ncg:f.ncg_smoothness,dirac_gap:f.dirac_gap,braid_len:f.braid_reduced_length,absorbance_rgb:f.absorbance_rgb,electrolyte:f.electrolyte||{}},null,2); }
async function poll(){ try{ const r=await fetch('frames_live.json?ts='+Date.now()); frames=await r.json(); draw(frames[frames.length-1]); }catch(e){ document.getElementById('status').innerText='waiting for frames_live.json...'; }}
setInterval(poll, 1000); poll();
</script></body></html>"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=150)
    ap.add_argument('--n_water',type=int,default=30)
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
    ap.add_argument('--preset',choices=['nacl','polar','hydrophobic','dye','photo','mixed'],default='mixed')
    ap.add_argument('--mode',choices=['boltzmann','random'],default='boltzmann')
    ap.add_argument('--temperature',type=float,default=0.55)
    ap.add_argument('--boltzmann_candidates', type=int, default=48, help='candidate pair budget for faster Boltzmann visual mode')
    ap.add_argument('--lambda_ncg',type=float,default=0.03)
    ap.add_argument('--lambda_gap',type=float,default=0.01)
    ap.add_argument('--lambda_braid',type=float,default=0.006)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--out_dir',type=str,default='runs/visual_chem_demo')
    ap.add_argument('--progress_every', type=int, default=10, help='print CLI progress every N steps')
    ap.add_argument('--live', action='store_true', help='write frames_live.json/dashboard.html during the run')
    ap.add_argument('--flush_every', type=int, default=5, help='flush live JSON every N steps')
    args=ap.parse_args(); random.seed(args.seed)
    if args.use_debye_huckel and args.electrolyte_model == 'none':
        args.electrolyte_model = 'dh' 
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    if args.use_physical_params:
        (out/'physical_params_report.json').write_text(json.dumps(default_parameter_report(args.temperature_K, args.dielectric), indent=2))
    if args.electrolyte_model != 'none' or args.use_onsager:
        es = ElectrolyteSettings(model=args.electrolyte_model, use_onsager=args.use_onsager, box_volume_l=args.box_volume_l, temperature_K=args.temperature_K, dielectric=args.dielectric, ion_size_nm=args.ion_size_nm, pitzer_beta0=args.pitzer_beta0, pitzer_beta1=args.pitzer_beta1, pitzer_cphi=args.pitzer_cphi)
        (out/'electrolyte_params_report.json').write_text(json.dumps(settings_report(es), indent=2))
    env=ChemHBondEnv(_make_cfg(args))
    braid=BraidTracker(len(env.nodes)); frames=[]
    eta_tracker = ETA(args.steps)
    if args.live:
        (out/'dashboard.html').write_text(live_dashboard_html())
        (out/'frames_live.json').write_text('[]')
        print('live dashboard:', out/'dashboard.html')
        print('serve with:  python3 -m http.server 8000 --directory', out)
    with MetricsJsonlWriter(out / "metrics_bundle.jsonl") as metrics_out:
        for step in range(args.steps):
            action=random.choice(env.possible_actions()) if args.mode=='random' else pick_boltzmann(env,args.temperature,args.boltzmann_candidates)
            st,base_r,_=env.step(action); braid.add_action(st['positions'],action,step=step)
            ncg_corr,ncg=ncg_reward_correction(st,args.lambda_ncg,args.lambda_gap); bm=braid.metrics(); b_corr=braid_reward_correction(bm,args.lambda_braid,0.001,0.0)
            pm = eta_tracker.tick(step)
            frame={'step':step,'action':f'{action[0]}({action[1]},{action[2]})','n_nodes':len(st['species']),'positions':st['positions'].tolist(),'types':st['species'], 'edges':[[i,j,float(w)] for (i,j),w in st['edges'].items()], 'energy':float(st['energy']),'energy_terms':st.get('energy_terms',{}),'energy_terms_kj_mol':st.get('energy_terms_kj_mol',{}),'energy_kj_mol':st.get('energy_kj_mol'),'electrolyte':st.get('electrolyte',{}),'reward':float(base_r+ncg_corr+b_corr),'species_counts':st['species_counts'],'absorbance':st.get('absorbance'),'absorbance_spectrum':st.get('absorbance_spectrum',{}),'absorbance_rgb':st.get('absorbance_rgb',[st['absorbance'], st['absorbance'], st['absorbance']]),'ncg_smoothness':ncg.ncg_smoothness,'dirac_gap':ncg.spectral_gap,'laplacian_gap':ncg.laplacian_gap,'braid_reduced_length':bm.reduced_length,'braid_writhe':bm.writhe,'n_edges':len(st['edges']),'braid_events':braid.events[-220:], **pm}
            frames.append(frame)
            metrics_out.write(frame)
            if args.progress_every and (step % args.progress_every == 0 or step + 1 == args.steps):
                print_progress(step, args.steps, pm, prefix='visual ')
            if args.live and (step % args.flush_every == 0 or step + 1 == args.steps):
                (out/'frames_live.json').write_text(json.dumps(frames))
    palette={k:v.color for k,v in SPECIES.items()}
    (out/'frames.json').write_text(json.dumps(frames,indent=2))
    (out/'visualizer.html').write_text(html(frames,palette))
    if args.auto_png:
        figs = generate_png_report(out, frames)
        print('wrote PNG figures:', out/'figures', f'({len(figs)} files)')
    print('wrote',out/'visualizer.html', 'and', out/'metrics_bundle.jsonl')
if __name__=='__main__': main()
