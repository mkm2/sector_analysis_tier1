"""
R32: search for a (time-reversal-type) symmetry W of the ring Floquet operator.
W = D L R^a T_s with D an arbitrary diagonal phase operator (solved exactly on the
support graph), L in {1, uniform/sublattice X or H}, R the site reflection, T_s the
translation; tests W U W^-1 = U^{+-1}, or with OMEGA=1 the half step Omega_T = T1 L_A.
Usage (from code/):  [OMEGA=1] python -m qca_fragmentation.spectral.symsearch 8 147 153
"""
import numpy as np, itertools, sys, os
from qca_fragmentation.core import rules, cycle
from collections import deque
from qca_fragmentation.spectral import floquet as F
N=int(sys.argv[1]); n=1<<N
OMEGA=os.environ.get('OMEGA')=='1'
def perm_op(f):
    P=np.zeros((n,n))
    for x in range(n):
        y=0
        for i in range(N):
            if (x>>i)&1: y|=1<<f(i)
        P[y,x]=1
    return P
I1=np.eye(2); X1=np.array([[0,1],[1,0]]); Z1=np.diag([1.,-1]); H1=np.array([[1,1],[1,-1]])/np.sqrt(2)
def onsite(g,pat):
    M=np.ones((1,1))
    for i in range(N): M=np.kron(g if pat(i) else I1, M)
    return M
locs={'1':np.eye(n)}
for nm,g in (('X',X1),('H',H1)):
    for pn,pat in (('all',lambda i:True),('even',lambda i:i%2==0),('odd',lambda i:i%2==1)):
        locs[nm+pn]=onsite(g,pat)
def layer(w, parity):
    t=rules.wolfram_to_tuple(w); L=np.eye(n)
    h=np.array([[1,1],[1,-1]])/np.sqrt(2)
    for i in range(parity,N,2):
        G=np.zeros((n,n))
        for x in range(n):
            m,nn=cycle.neighbor_bits(x,i,N,'pbc')
            if t[2*m+nn]=='V':
                b=(x>>i)&1
                for c in (0,1): G[(x&~(1<<i))|(c<<i),x]+=h[c,b]
            else: G[x,x]=1
        L=G@L
    return L
def layerA(w):
    LA=layer(w,0); LB=layer(w,1)
    assert np.abs(LB@LA-F.floquet_matrix(w,N,'H','pbc')).max()<1e-9, 'layer convention'
    return LA
def solve(M,T,tol=1e-9):
    # find unit-modulus d with d_x M_xy conj(d_y) = T_xy  (D M D^dag = T)
    if np.any((np.abs(M)>tol)!=(np.abs(T)>tol)): return None
    d=np.full(n,np.nan+0j); 
    for s0 in range(n):
        if not np.isnan(d[s0]): continue
        d[s0]=1; q=deque([s0])
        while q:
            x=q.popleft()
            for y in np.nonzero(np.abs(M[x])>tol)[0]:
                v=T[x,y]/(M[x,y]); dy=np.conj(v)*d[x]   # d_x M conj(d_y)=T -> conj(d_y)=T/(M d_x)
                dy=np.conj(T[x,y]/(M[x,y]*d[x]))
                if np.isnan(d[y]): d[y]=dy; q.append(y)
                elif abs(d[y]-dy)>1e-7: return None
            for y in np.nonzero(np.abs(M[:,x])>tol)[0]:
                dy=T[y,x]/(M[y,x])*d[x]   # d_y M_yx conj(d_x)=T_yx
                if np.isnan(d[y]): d[y]=dy; q.append(y)
                elif abs(d[y]-dy)>1e-7: return None
    return d
for w in [int(a) for a in sys.argv[2:]]:
    U=F.floquet_matrix(w,N,"H","pbc").astype(complex)
    if OMEGA:
        LA=layerA(w); U=perm_op(lambda i:(i+1)%N)@LA
        assert np.abs(U@U - F.floquet_matrix(w,N,"H","pbc")@perm_op(lambda i:(i+2)%N)).max()<1e-9 or np.abs(U@U - F.floquet_matrix(w,N,"H","pbc")@perm_op(lambda i:(i-2)%N)).max()<1e-9
    Ui=U.conj().T
    hits=[]
    for s in range(N):
        RT=perm_op(lambda i,s=s:(-i+s)%N)
        for ln,L in locs.items():
            W=L@RT; M=W@U@W.conj().T
            for tn,T in (('U^-1',Ui),('U',U)):
                d=solve(M,T)
                if d is not None:
                    hits.append(f'D*{ln}*R.T{s} -> {tn}; D phases: {sorted(set(np.round(np.angle(d)/np.pi,3)))[:6]}')
    print(w,hits if hits else 'none')
