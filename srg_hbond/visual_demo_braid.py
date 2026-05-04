from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

from .env import HBondGraphEnv, SRGConfig
from .ncg import ncg_reward_correction
from .braid import BraidTracker, braid_reward_correction


def pick_boltzmann(env, temperature=0.5):
    acts = env.possible_actions(limit_pairs=240)
    e0 = env.free_energy(); ws = []
    for a in acts:
        c = env.clone(); c.step(a)
        de = c.free_energy() - e0
        ws.append(math.exp(-de / max(temperature, 1e-6)))
    s = sum(ws); r = random.random() * s; acc = 0
    for a, w in zip(acts, ws):
        acc += w
        if acc >= r: return a
    return acts[-1]


def html(frames):
    payload = json.dumps(frames)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>SRG-NCG-Braid Visualizer</title>
<style>
body{{font-family:system-ui,Arial;margin:20px;background:#101218;color:#f1f1f1}} .wrap{{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}} canvas{{background:#fff;border-radius:14px}} .panel{{min-width:360px;background:#191d29;padding:16px;border-radius:14px}} input{{width:100%}} code{{color:#9ee}} .small{{opacity:.8;font-size:13px}}
</style></head><body>
<h1>SRG-NCG-Braid H-bond Visualizer</h1>
<div class='wrap'>
<canvas id='cv' width='650' height='650'></canvas>
<canvas id='bcv' width='420' height='650'></canvas>
<div class='panel'>
<button id='play'>Play</button><input id='slider' type='range' min='0' max='{len(frames)-1}' value='0'/>
<pre id='info'></pre>
<p><b>Legend:</b> red = solute, blue = solvent, gray = H-bond/solvation edge.</p>
<p class='small'>Braid panel: vertical axis=time, horizontal axis=generator index; green=σᵢ, orange=σᵢ⁻¹. The braid word is a topology feature extracted from form/break events.</p>
</div></div>
<script>
const frames={payload}; const cv=document.getElementById('cv'); const ctx=cv.getContext('2d');
const bcv=document.getElementById('bcv'); const bctx=bcv.getContext('2d');
const slider=document.getElementById('slider'); const info=document.getElementById('info'); let playing=false;
function drawBraid(k){{
  bctx.clearRect(0,0,420,650); bctx.fillStyle='#fff'; bctx.fillRect(0,0,420,650);
  const events=frames[k].braid_events; const n=frames[k].n_nodes; const maxGen=Math.max(1,n-1);
  bctx.strokeStyle='rgba(0,0,0,.12)'; bctx.lineWidth=1;
  for(let g=1; g<=Math.min(maxGen,20); g++){{ const x=30+(g-1)*360/Math.max(1,Math.min(maxGen,20)-1); bctx.beginPath(); bctx.moveTo(x,20); bctx.lineTo(x,630); bctx.stroke(); }}
  for(let idx=0; idx<events.length; idx++){{ const e=events[idx]; const x=30+(Math.min(e.gen,20)-1)*360/Math.max(1,Math.min(maxGen,20)-1); const y=30+idx*590/Math.max(1,events.length-1); bctx.beginPath(); bctx.arc(x,y,5,0,Math.PI*2); bctx.fillStyle=e.dir>0?'#19a974':'#ff8c00'; bctx.fill(); }}
  bctx.fillStyle='#111'; bctx.font='13px system-ui'; bctx.fillText('Braid events over time',20,18);
}}
function draw(k){{ const f=frames[k]; ctx.clearRect(0,0,650,650); const P=f.positions;
  ctx.lineWidth=2; ctx.strokeStyle='rgba(80,80,80,0.55)';
  for(const e of f.edges){{ const a=P[e[0]], b=P[e[1]], w=e[2]; ctx.globalAlpha=Math.min(1,0.2+w); ctx.beginPath(); ctx.moveTo(35+a[0]*580,35+a[1]*580); ctx.lineTo(35+b[0]*580,35+b[1]*580); ctx.stroke(); }} ctx.globalAlpha=1;
  for(let i=0;i<P.length;i++){{ const p=P[i]; ctx.beginPath(); ctx.arc(35+p[0]*580,35+p[1]*580,i==0?13:7,0,Math.PI*2); ctx.fillStyle=i==0?'#d43':'#48f'; ctx.fill(); ctx.strokeStyle='#111'; ctx.stroke(); }}
  info.textContent = `step: ${{f.step}}\naction: ${{f.action}}\nenergy: ${{f.energy.toFixed(4)}}\nreward: ${{f.reward.toFixed(4)}}\nNCG smoothness: ${{f.ncg_smoothness.toFixed(4)}}\nDirac gap: ${{f.dirac_gap.toFixed(4)}}\nLaplacian gap: ${{f.laplacian_gap.toFixed(4)}}\nedges: ${{f.n_edges}}\nsolute degree: ${{f.solvation_shell_degree}}\nbraid raw/reduced: ${{f.braid_raw_length}} / ${{f.braid_reduced_length}}\nwrithe: ${{f.braid_writhe}}\nbraid entropy: ${{f.braid_entropy.toFixed(4)}}\nword: ${{f.braid_word}}`;
  drawBraid(k);
}}
slider.oninput=()=>draw(+slider.value); document.getElementById('play').onclick=()=>{{playing=!playing}};
setInterval(()=>{{ if(playing){{ slider.value=(+slider.value+1)%frames.length; draw(+slider.value); }} }},140); draw(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--n_solvent', type=int, default=30)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--lambda_ncg', type=float, default=0.05)
    ap.add_argument('--lambda_gap', type=float, default=0.01)
    ap.add_argument('--lambda_braid', type=float, default=0.01)
    ap.add_argument('--lambda_writhe', type=float, default=0.002)
    ap.add_argument('--lambda_braid_entropy', type=float, default=0.0)
    ap.add_argument('--mode', choices=['boltzmann','random'], default='boltzmann')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='runs/visual_braid_demo')
    args = ap.parse_args()
    random.seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    env=HBondGraphEnv(SRGConfig(n_solvent=args.n_solvent, seed=args.seed))
    braid=BraidTracker(env.n_nodes)
    frames=[]
    for step in range(args.steps):
        action = random.choice(env.possible_actions(limit_pairs=240)) if args.mode == 'random' else pick_boltzmann(env,args.temperature)
        st, reward, _ = env.step(action)
        braid.add_action(st['positions'], action, step=step)
        corr_ncg, m = ncg_reward_correction(st,args.lambda_ncg,args.lambda_gap)
        corr_braid, bm = braid_reward_correction(braid.metrics(),args.lambda_braid,args.lambda_writhe,args.lambda_braid_entropy)
        frames.append({
            'step': step, 'action': f'{action[0]}({action[1]},{action[2]})', 'n_nodes': st['n_nodes'],
            'positions': st['positions'].tolist(), 'edges': [[i,j,float(w)] for (i,j),w in st['edges'].items()],
            'energy': float(st['energy']), 'reward': float(reward+corr_ncg+corr_braid),
            'ncg_smoothness': m.ncg_smoothness, 'dirac_gap': m.spectral_gap, 'dirac_radius': m.dirac_radius,
            'laplacian_gap': m.laplacian_gap, 'n_edges': m.n_edges, 'solvation_shell_degree': int(st['solvation_shell_degree']),
            'braid_raw_length': bm.raw_length, 'braid_reduced_length': bm.reduced_length, 'braid_writhe': bm.writhe,
            'braid_entropy': bm.generator_entropy, 'braid_permutation_disorder': bm.permutation_disorder,
            'braid_word': braid.word_string(), 'braid_events': braid.events[-200:],
        })
    with open(out/'frames.json','w') as f: json.dump(frames,f)
    with open(out/'visualizer.html','w') as f: f.write(html(frames))
    print('wrote', out/'visualizer.html')
if __name__=='__main__': main()
