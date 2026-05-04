from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class NCGMetrics:
    ncg_smoothness: float
    spectral_gap: float
    dirac_radius: float
    laplacian_gap: float
    n_edges: int

def incidence_dirac(n_nodes, edges):
    m=len(edges); B=np.zeros((n_nodes,m),dtype=np.float64)
    for k,((i,j),w) in enumerate(edges.items()):
        s=float(max(w,1e-8))**0.5; B[i,k]=s; B[j,k]=-s
    D=np.block([[np.zeros((n_nodes,n_nodes)),B],[B.T,np.zeros((m,m))]]) if m else np.zeros((n_nodes,n_nodes))
    return D,B

def _node_labels(state):
    if 'types' in state: return state['types']
    if 'species' in state: return state['species']
    return ['solute' if i == 0 else 'water' for i in range(state.get('n_nodes', len(state.get('adjacency', []))))]

def spectral_metrics(state):
    labels = _node_labels(state)
    n = len(labels)
    edges=state['edges']; D,B=incidence_dirac(n,edges)
    ev=np.linalg.eigvalsh(D) if D.size else np.zeros(1)
    pos=np.sort(np.abs(ev[np.abs(ev)>1e-8])); gap=float(pos[0]) if len(pos) else 0.0
    radius=float(np.max(np.abs(ev))) if len(ev) else 0.0
    if B.size:
        L=B@B.T; lev=np.linalg.eigvalsh(L); lap=float(sorted(lev)[1]) if len(lev)>1 else 0.0
    else: lap=0.0
    # f is a chemical/spectral observable on nodes. Smoothness = graph Dirichlet energy.
    amap={'water':0.05,'solute':0.25,'na':0.02,'cl':0.02,'Na+':0.02,'Cl-':0.02,
          'polar':0.35,'hydrophobic':0.15,'dye':0.75,'photo_dye':0.90,'photo_product':0.55,
          'dye_A':0.85,'dye_B':0.55}
    vals=[amap.get(s,0.1) for s in labels]
    smooth=0.0
    for (i,j),w in edges.items(): smooth += float(w*w*(vals[i]-vals[j])**2)
    return NCGMetrics(float(smooth),gap,radius,lap,len(edges))

def ncg_reward_correction(state, lambda_ncg=0.03, lambda_gap=0.0):
    m=spectral_metrics(state)
    return -lambda_ncg*m.ncg_smoothness + lambda_gap*m.laplacian_gap, m
