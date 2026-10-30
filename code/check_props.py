"""Numerical checks of Propositions 1-2 and Lemma 1 on 1,999 MNIST digits (results/concept). usage: python3 code/check_props.py"""
import numpy as np
from scipy.spatial.distance import cdist
z=np.load('results/concept/concept_mnist70k.npz'); S=z['S']; X=z['XS'].astype(float)
X=X[:1999]; S=S[:1999]; n=len(X)
def ranks(D):
    D=D.copy(); np.fill_diagonal(D,np.inf); o=np.argsort(D,1,kind='stable'); R=np.empty_like(o); R[np.arange(n)[:,None],o]=np.arange(1,n+1)[None,:]; return R  # self gets rank n
RX=ranks(cdist(X,X))
def prof(RY,ks):
    out={}; per={}
    for k in ks:
        O=((RX<=k)&(RY<=k)).sum(1)
        nui=((n-1)*O/k-k)/(n-1-k); per[k]=nui; out[k]=((n-1)*O.sum()/(n*k)-k)/(n-1-k)
    return out,per
ks=[10,100,300,600,800,999]
rng=np.random.default_rng(0)
rv=[]
for r in range(20):
    Y=rng.standard_normal((n,2)); o,_=prof(ranks(cdist(Y,Y)),ks); rv.append(list(o.values()))
rv=np.array(rv); print('random map nu: mean',rv.mean(),'max abs mean over runs per k',np.abs(rv.mean(0)).max(),'sd',rv.std())
ks2=999
for m in ['PCA','t_SNE','SR_tSNE2']:
    Y=z[m][S].astype(float); DY=cdist(Y,Y); RY=ranks(DY)
    o,per=prof(RY,[10,ks2])
    # monotone transform
    RY3=ranks(DY**3+np.exp(DY)); o3,_=prof(RY3,[10,ks2])
    agree=((RX<=ks2)==(RY<=ks2)); np.fill_diagonal(agree,False); h=agree.sum(1)/(n-1)
    print(m,'nu10',o[10],'mean nu10(i)',per[10].mean(),'diff',abs(o[10]-per[10].mean()),'nu_k*',o[ks2],'2h-1',2*h.mean()-1,'diff',abs(o[ks2]-(2*h.mean()-1)),'mono diff',abs(o[10]-o3[10])+abs(o[ks2]-o3[ks2]))
