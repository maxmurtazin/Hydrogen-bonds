from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

from .env import HBondGraphEnv, SRGConfig
from .ncg import ncg_reward_correction, spectral_metrics


def pick_boltzmann(env, temperature=0.5):
    acts = env.possible_actions(limit_pairs=240)
    e0 = env.free_energy()
    ws = []
    for a in acts:
        c = env.clone(); c.step(a)
        de = c.free_energy() - e0
        ws.append(math.exp(-de / max(temperature, 1e-6)))
    s = sum(ws)
    r = random.random() * s
    acc = 0
    for a, w in zip(acts, ws):
        acc += w
        if acc >= r:
            return a
    return acts[-1]


def html(frames):
    payload = json.dumps(frames)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'/><title>SRG-NCG H-bond Visualizer</title>
<style>
body{{font-family:system-ui,Arial;margin:20px;background:#101218;color:#f1f1f1}} .wrap{{display:flex;gap:20px;align-items:flex-start}} canvas{{background:#fff;border-radius:14px}} .panel{{min-width:340px;background:#191d29;padding:16px;border-radius:14px}} input{{width:100%}} code{{color:#9ee}}
</style></head><body>
<h1>SRG-NCG H-bond Graph Visualizer</h1>
<div class='wrap'><canvas id='cv' width='720' height='720'></canvas><div class='panel'>
<button id='play'>Play</button><input id='slider' type='range' min='0' max='{len(frames)-1}' value='0'/>
<pre id='info'></pre>
<p><b>Legend:</b> red = solute, blue = solvent, gray = H-bond/solvation edge.</p>
<p>NCG metrics come from graph Dirac operator <code>D=[[0,B],[B^T,0]]</code>.</p>
</div></div>
<script>
const frames={payload}; const cv=document.getElementById('cv'); const ctx=cv.getContext('2d');
const slider=document.getElementById('slider'); const info=document.getElementById('info'); let playing=false;
function draw(k){{ const f=frames[k]; ctx.clearRect(0,0,720,720); const P=f.positions; 
  ctx.lineWidth=2; ctx.strokeStyle='rgba(80,80,80,0.55)';
  for(const e of f.edges){{ const a=P[e[0]], b=P[e[1]], w=e[2]; ctx.globalAlpha=Math.min(1,0.2+w); ctx.beginPath(); ctx.moveTo(40+a[0]*640,40+a[1]*640); ctx.lineTo(40+b[0]*640,40+b[1]*640); ctx.stroke(); }} ctx.globalAlpha=1;
  for(let i=0;i<P.length;i++){{ const p=P[i]; ctx.beginPath(); ctx.arc(40+p[0]*640,40+p[1]*640,i==0?13:7,0,Math.PI*2); ctx.fillStyle=i==0?'#d43':'#48f'; ctx.fill(); ctx.strokeStyle='#111'; ctx.stroke(); }}
  info.textContent = `step: ${{f.step}}\naction: ${{f.action}}\nenergy: ${{f.energy.toFixed(4)}}\nreward: ${{f.reward.toFixed(4)}}\nNCG smoothness: ${{f.ncg_smoothness.toFixed(4)}}\nDirac gap: ${{f.dirac_gap.toFixed(4)}}\nDirac radius: ${{f.dirac_radius.toFixed(4)}}\nLaplacian gap: ${{f.laplacian_gap.toFixed(4)}}\nedges: ${{f.n_edges}}\nsolute degree: ${{f.solvation_shell_degree}}`;
}}
slider.oninput=()=>draw(+slider.value); document.getElementById('play').onclick=()=>{{playing=!playing}};
setInterval(()=>{{ if(playing){{ slider.value=(+slider.value+1)%frames.length; draw(+slider.value); }} }},120); draw(0);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--n_solvent', type=int, default=30)
    ap.add_argument('--temperature', type=float, default=0.5)
    ap.add_argument('--lambda_ncg', type=float, default=0.05)
    ap.add_argument('--lambda_gap', type=float, default=0.01)
    ap.add_argument('--mode', choices=['boltzmann','random'], default='boltzmann')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='runs/visual_ncg_demo')
    args = ap.parse_args()
    random.seed(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    env=HBondGraphEnv(SRGConfig(n_solvent=args.n_solvent, seed=args.seed))
    frames=[]
    for step in range(args.steps):
        if args.mode == 'random': action=random.choice(env.possible_actions(limit_pairs=240))
        else: action=pick_boltzmann(env,args.temperature)
        st, reward, info = env.step(action)
        corr, m = ncg_reward_correction(st,args.lambda_ncg,args.lambda_gap)
        frames.append({
            'step': step, 'action': f'{action[0]}({action[1]},{action[2]})',
            'positions': st['positions'].tolist(),
            'edges': [[i,j,float(w)] for (i,j),w in st['edges'].items()],
            'energy': float(st['energy']), 'reward': float(reward+corr),
            'ncg_smoothness': m.ncg_smoothness, 'dirac_gap': m.spectral_gap,
            'dirac_radius': m.dirac_radius, 'laplacian_gap': m.laplacian_gap,
            'n_edges': m.n_edges, 'solvation_shell_degree': int(st['solvation_shell_degree'])})
    with open(out/'frames.json','w') as f: json.dump(frames,f)
    with open(out/'visualizer.html','w') as f: f.write(html(frames))
    print('wrote', out/'visualizer.html')
if __name__=='__main__': main()
