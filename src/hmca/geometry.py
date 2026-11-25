"""Actor history to lane segments and local robust SE(2) refinement."""
import math
import numpy as np
from scipy.spatial import cKDTree

# Optimize in normalized coordinates: 2 m, 2 m, 2 degrees.
SCALE = np.array([2.,2.,math.radians(2)])
S_YAW = math.radians(10)
BOUND = 1.5
LATERAL_SCALE = .75


def wrap(a):
    return np.arctan2(np.sin(a),np.cos(a))


def tracks(positions,headings,padding):
    # Temporal spacing avoids letting adjacent repeated samples dominate.
    times=np.arange(4,50,5)
    prev=times-4
    d=positions[:,times]-positions[:,prev]
    speed=np.linalg.norm(d,axis=-1)
    a=headings[:,times]
    alignment=np.cos(wrap(np.arctan2(d[...,1],d[...,0])-a))
    valid=(~padding[:,times]) & (~padding[:,prev]) & (~padding[:,-1,None])
    weight=np.where(valid,np.minimum(speed/.8,1.)*np.maximum(alignment,0.)**2,0.)
    actor_weight=weight.sum(1,keepdims=True)
    weight/=np.maximum(actor_weight,1.)
    selected=weight>1e-6
    if not selected.any():
        return np.empty((0,2)),np.empty(0),np.empty(0),0
    return positions[:,times][selected],a[selected],weight[selected]/weight.sum(),int(selected.any(1).sum())


class Geometry:
    def __init__(self,query,headings,weights,lanes,pad):
        p,q=lanes[:,:-1],lanes[:,1:]
        size=np.linalg.norm(q-p,axis=-1)
        keep=(~pad[:,:-1]) & (~pad[:,1:]) & (size>1e-5)
        self.origins=p[keep]
        self.size=size[keep]
        self.tangent=(q-p)[keep]/self.size[:,None]
        self.normal=self.tangent[:,[1,0]]*np.array([-1,1])
        self.lane_heading=np.arctan2(self.tangent[:,1],self.tangent[:,0])
        self.query=query
        self.headings=headings
        self.weights=weights
        self.available=len(self.origins)>0 and len(query)>=3
        if self.available:
            self.tree=cKDTree(np.concatenate([self.origins,(p+q)[keep]*.5,q[keep]]))

    def model(self,z):
        t=z*SCALE
        c,s=np.cos(t[2]),np.sin(t[2])
        src=self.query@np.array([[c,s],[-s,c]])
        return src,src+t[:2]

    def associate(self,z):
        _,q=self.model(z)
        _,idx=self.tree.query(q,k=min(24,self.tree.n))
        if idx.ndim==1:
            idx=idx[:,None]
        idx=idx%len(self.origins)
        offset=q[:,None]-self.origins[idx]
        u=self.tangent[idx]
        along=(offset*u).sum(-1)
        distance=np.linalg.norm(offset-np.clip(along,0,self.size[idx])[...,None]*u,axis=-1)
        e=wrap(self.headings[:,None]+z[2]*SCALE[2]-self.lane_heading[idx])
        valid=(distance<6)&(np.abs(e)<math.radians(60))
        for i in range(1,idx.shape[1]):
            valid[:,i]&=~(idx[:,i,None]==idx[:,:i]).any(1)
        score=-.5*((distance/1.75)**2+(e/math.radians(20))**2)
        likelihood=np.exp(np.where(valid,score,-1e4)-np.max(np.where(valid,score,-1e4),axis=1,keepdims=True))*valid
        prob=likelihood/np.maximum(likelihood.sum(1,keepdims=True),1e-12)
        return idx,prob,distance

    def evaluate(self,z,assoc):
        idx,prob,_=assoc
        src,q=self.model(z)
        n=self.normal[idx]
        lateral=((q[:,None]-self.origins[idx])*n).sum(-1)/LATERAL_SCALE
        yaw=wrap(self.headings[:,None]+z[2]*SCALE[2]-self.lane_heading[idx])/S_YAW
        dtheta=src[:,[1,0]]*np.array([-1,1])*SCALE[2]
        j=np.concatenate((n*SCALE[None,None,:2],(n*dtheta[:,None]).sum(-1)[...,None]),axis=-1)/LATERAL_SCALE
        jy=SCALE[2]/S_YAW
        w=self.weights[:,None]*prob
        robust=lambda r: 1/np.maximum(1.,np.abs(r))
        wlat=w*robust(lateral)
        wyaw=w*robust(yaw)
        H=np.einsum('nki,nkj,nk->ij',j,j,wlat)
        g=np.einsum('nki,nk,nk->i',j,wlat,lateral)
        H[2,2]+=wyaw.sum()*jy**2
        g[2]+=(wyaw*yaw).sum()*jy
        huber=lambda r: np.where(np.abs(r)<=1,.5*r*r,np.abs(r)-.5)
        support=float(w.sum())
        cost=float((w*(huber(lateral)+huber(yaw))).sum()+1.5*(1-support))
        return H,g,cost,support


def refine(geo,initial,observable):
    z0=np.clip(np.asarray(initial)/SCALE,-BOUND,BOUND)
    diagnostics=dict(status='insufficient_history_or_map',support=0.,eig=[0.]*3,
                     translation_eig=[0.]*2,normal_rank=0,ambiguous=1.,accepted_steps=0)
    if not geo.available:
        return np.asarray(initial),diagnostics
    z=z0.copy()
    for step in range(5):
        assoc=geo.associate(z)
        H,g,_,support=geo.evaluate(z,assoc)
        idx,prob,_=assoc
        # Nearby samples from a common corridor are not independent lane hypotheses.
        # Only substantial direction disagreement reduces confidence in a match.
        td=geo.tangent[idx]
        consensus=np.linalg.norm((prob[...,None]*td).sum(1),axis=-1)
        confidence=consensus**2 if observable else np.ones_like(consensus)
        H,g,cost,support=geo.evaluate(z,assoc)
        if observable:
            weighted=(idx,prob*confidence[:,None],assoc[2])
            H,g,cost,support=geo.evaluate(z,weighted)
        eig,U=np.linalg.eigh(H)
        schur=H[:2,:2]-np.outer(H[:2,2],H[2,:2])/max(H[2,2],1e-9)
        ts=np.linalg.eigvalsh(schur)
        diagnostics.update(status='insufficient_support' if support<.45 else 'finite',
                           support=support,eig=eig.tolist(),translation_eig=ts.tolist(),
                           normal_rank=int((eig>1e-6).sum()),
                           ambiguous=float((geo.weights*(1-consensus)).sum()))
        if support<.45:
            break
        # A proposal cannot invent a new correction in the nullspace.
        threshold=.07 if observable else .004
        inv=1/(np.maximum(eig,0)+threshold)
        if observable:
            inv*=np.maximum(eig,0)/(np.maximum(eig,0)+threshold)
        ridge=.08
        update=-U@(inv*(U.T@(g+ridge*(z-z0))))
        old=cost+.5*ridge*((z-z0)**2).sum()
        found=False
        for fraction in (1.,.5,.25,.125):
            candidate=np.clip(z+fraction*update,-BOUND,BOUND)
            candidate_cost=geo.evaluate(candidate,weighted if observable else assoc)[2]
            candidate_cost+=.5*ridge*((candidate-z0)**2).sum()
            if candidate_cost<old-1e-7:
                z=candidate
                found=True
                diagnostics['accepted_steps']+=1
                break
        if not found or np.linalg.norm(update)<1e-4:
            break
    return z*SCALE,diagnostics


def synthetic():
    x=np.linspace(-25,25,41)
    first=np.column_stack((x,x*0))
    second=np.column_stack((x*0,x))
    q=np.concatenate([first[8:32],second[8:32]])
    angles=np.r_[np.zeros(24),np.full(24,np.pi/2)]
    g1=Geometry(q[:24],angles[:24],np.full(24,1/24),first[None],np.zeros((1,41),bool))
    g2=Geometry(q,angles,np.full(48,1/48),np.stack([first,second]),np.zeros((2,41),bool))
    h1=g1.evaluate(np.zeros(3),g1.associate(np.zeros(3)))[0]
    h2=g2.evaluate(np.zeros(3),g2.associate(np.zeros(3)))[0]
    shift=np.array([.4,0,0])
    inv=g1.evaluate(shift,g1.associate(shift))[2]
    zero=g1.evaluate(np.zeros(3),g1.associate(np.zeros(3)))[2]
    assert np.linalg.eigvalsh(h1)[0]<1e-10 and np.linalg.eigvalsh(h2)[0]>.01
    assert abs(inv-zero)<1e-12
    return dict(parallel_eigenvalues=np.linalg.eigvalsh(h1).tolist(),
                crossing_eigenvalues=np.linalg.eigvalsh(h2).tolist(),
                longitudinal_cost_change=float(inv-zero))


def solve_scene(positions,headings,padding,lanes,lane_pad,initial):
    p,a,w,n=tracks(positions,headings,padding)
    geo=Geometry(p,a,w,lanes,lane_pad)
    a,diag=refine(geo,initial,False)
    b,obs=refine(geo,initial,True)
    obs['moving_tracks']=n
    return a,b,diag,obs
