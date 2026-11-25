"""Single-scene RALR geometric-registration baseline."""

import math
import time

import numpy as np
from scipy.spatial import cKDTree


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def transform(p, delta):
    x,y,deg=delta; a=np.deg2rad(deg); c,s=np.cos(a),np.sin(a)
    return p @ np.array([[c,s],[-s,c]]) + np.array([x,y])

def huber_cost(r, delta=1.0):
    a=np.abs(r)
    return np.where(a<=delta,0.5*r*r,delta*(a-0.5*delta))

class Solver:
    def __init__(self,config): self.cfg=config

    def solve(self,h,ha,hmask,all_h,all_ha,all_mask,lanes,lmask,lkey):
        start_time=time.perf_counter(); cfg=self.cfg
        def fallback(reason,**extra):
            return np.zeros(3),dict(status=reason,success=False,identity_fallback=True,
                                   elapsed_s=time.perf_counter()-start_time,**extra)
        # Exclude every segment incident to a padded endpoint or padded lane.
        valid=~(lmask[:,:-1]|lmask[:,1:]|lkey[:,None])
        segment_a=lanes[:,:-1][valid]; segment_b=lanes[:,1:][valid]
        length=np.linalg.norm(segment_b-segment_a,axis=1)
        ok=np.isfinite(segment_a).all(1)&np.isfinite(segment_b).all(1)&(length>1e-5)
        segment_a,segment_b,length=segment_a[ok],segment_b[ok],length[ok]
        if not len(segment_a): return fallback('no_valid_lane_segments')
        tangent=(segment_b-segment_a)/length[:,None]
        normal=np.stack([-tangent[:,1],tangent[:,0]],-1)
        heading=np.arctan2(tangent[:,1],tangent[:,0])
        midpoint=(segment_a+segment_b)*0.5
        # Two endpoints plus midpoint avoid rejecting long lane segments.
        anchor=np.concatenate([segment_a,midpoint,segment_b])
        segment_id=np.tile(np.arange(len(segment_a)),3)
        tree=cKDTree(anchor)
        hv=(~hmask)&np.isfinite(h).all(1)&np.isfinite(ha)
        av=(~all_mask[:,-1])&np.isfinite(all_h[:,-1]).all(1)&np.isfinite(all_ha[:,-1])
        nf,na=int(hv.sum()),int(av.sum())
        if nf+na<cfg['min_effective_observations']: return fallback('too_few_observations')
        query=np.concatenate([h[hv],all_h[av,-1]])
        angles=np.concatenate([ha[hv],all_ha[av,-1]])
        fw=0.5 if na and nf else 1.0; aw=0.5 if nf and na else 1.0
        weight=np.r_[np.full(nf,fw/max(nf,1)),np.full(na,aw/max(na,1))]
        k=min(cfg['candidate_segments'],len(anchor))
        yaw_scale=np.deg2rad(cfg['heading_scale_deg'])
        heading_temp=np.deg2rad(cfg['association_heading_temperature_deg'])
        heading_gate=np.deg2rad(cfg['association_heading_deg'])

        def associate(delta):
            q=transform(query,delta)
            _,idx=tree.query(q,k=k)
            idx=np.asarray(idx).reshape(len(q),k)
            ids=segment_id[idx]
            a=segment_a[ids]; u=tangent[ids]; n=normal[ids]
            offset=q[:,None]-a
            along=np.clip((offset*u).sum(-1),0,length[ids])
            closest=a+along[...,None]*u
            dist=np.linalg.norm(q[:,None]-closest,axis=-1)
            angle_error=wrap(angles[:,None]+np.deg2rad(delta[2])-heading[ids])
            accepted=(dist<=cfg['association_radius_m'])&(np.abs(angle_error)<=heading_gate)
            # Duplicate anchors for one segment must not multiply its influence.
            for col in range(1,k): accepted[:,col]&=~(ids[:,col,None]==ids[:,:col]).any(1)
            valid_query=accepted.any(1)
            exponent=-0.5*(dist/cfg['association_distance_temperature_m'])**2-0.5*(angle_error/heading_temp)**2
            exponent=np.where(accepted,exponent,-1e4)
            probs=np.exp(exponent-exponent.max(1,keepdims=True))*accepted
            probs/=np.maximum(probs.sum(1,keepdims=True),1e-12)
            return a,n,heading[ids],probs,valid_query

        def residual(delta,assoc):
            a,n,lane_heading,probs,valid_query=assoc
            q=transform(query,delta)
            lateral=((q[:,None]-a)*n).sum(-1)/cfg['lateral_scale_m']
            angular=wrap(angles[:,None]+np.deg2rad(delta[2])-lane_heading)/yaw_scale
            # Robustify residuals BEFORE weighting so outlier thresholds retain
            # their stated geometric scales rather than changing with actor count.
            lat=np.sign(lateral)*np.sqrt(2*huber_cost(lateral,cfg['huber_delta']))
            ang=np.sign(angular)*np.sqrt(2*huber_cost(angular,cfg['huber_delta']))
            w=np.sqrt(weight[:,None]*probs)
            prior=np.r_[np.sqrt(cfg['translation_prior_weight'])*np.asarray(delta[:2])/3,
                        np.sqrt(cfg['yaw_prior_weight'])*delta[2]/3]
            return np.r_[(lat*w).ravel(),(ang*w).ravel(),prior]

        def objective(delta):
            assoc=associate(delta)
            support=float(weight[assoc[-1]].sum())
            # Fixed penalty for unmatched observations prevents disappearing
            # correspondences from artificially improving the objective.
            return .5*np.sum(residual(delta,assoc)**2)+2*(1-support),assoc

        def optimize_fixed(delta,assoc):
            """Analytic Huber IRLS with box projection and monotone line search.

            This minimizes exactly the same fixed-correspondence Huber objective
            as the initial SciPy finite-difference prototype, without thousands
            of Python residual callbacks. Parameters are [tx(m),ty(m),yaw(deg)].
            """
            a,n,lane_heading,probs,_=assoc
            qweight=weight[:,None]*probs
            prior=np.array([cfg['translation_prior_weight']]*2+[cfg['yaw_prior_weight']])/9
            evaluations=0
            for inner in range(cfg['inner_max_nfev']):
                q=transform(query,delta)
                rotated=q-delta[None,:2]
                dyaw=np.stack([-rotated[:,1],rotated[:,0]],-1)*(math.pi/180)
                lateral=((q[:,None]-a)*n).sum(-1)/cfg['lateral_scale_m']
                angular=wrap(angles[:,None]+np.deg2rad(delta[2])-lane_heading)/yaw_scale
                J=np.concatenate([n,((n*dyaw[:,None]).sum(-1))[...,None]],axis=-1)/cfg['lateral_scale_m']
                wlat=qweight*np.minimum(1,cfg['huber_delta']/np.maximum(np.abs(lateral),1e-12))
                wang=qweight*np.minimum(1,cfg['huber_delta']/np.maximum(np.abs(angular),1e-12))
                H=np.einsum('qki,qkj,qk->ij',J,J,wlat)
                g=np.einsum('qki,qk,qk->i',J,lateral,wlat)
                jangle=math.pi/180/yaw_scale
                H[2,2]+=wang.sum()*jangle*jangle
                g[2]+=(wang*angular).sum()*jangle
                H+=np.diag(prior+1e-9);g+=prior*delta
                step=np.linalg.solve(H,-g)
                if np.linalg.norm(step)<1e-5:break
                old=.5*np.sum(residual(delta,assoc)**2);accepted=False
                for scale in [1,.5,.25,.125,.0625]:
                    candidate=np.clip(delta+scale*step,-bound,bound)
                    new=.5*np.sum(residual(candidate,assoc)**2);evaluations+=1
                    if new<=old+1e-12:
                        accepted=True;break
                if not accepted:break
                change=np.linalg.norm(candidate-delta);delta=candidate
                if change<1e-5 or abs(old-new)<1e-8:break
            return delta,evaluations

        base,base_assoc=objective(np.zeros(3))
        candidates=[]
        bound=np.array([cfg['max_translation_component_m']]*2+[cfg['max_yaw_deg']])
        for initial in cfg['starts']:
            delta=np.array(initial,dtype=float); converged=False; evaluations=0
            for iteration in range(cfg['outer_iterations']):
                assoc=associate(delta)
                if assoc[-1].sum()<cfg['min_effective_observations']: break
                solved,evaluated=optimize_fixed(delta,assoc)
                evaluations+=evaluated
                if not np.isfinite(solved).all(): break
                change=np.linalg.norm(solved-delta);delta=solved
                if change<1e-4: converged=True;break
            cost,assoc=objective(delta)
            if np.isfinite(cost) and assoc[-1].sum()>=cfg['min_effective_observations']:
                candidates.append((cost,delta,assoc,converged,evaluations))
        if not candidates: return fallback('association_failure',focal_observations=nf,current_actors=na)
        cost,delta,assoc,converged,evaluations=min(candidates,key=lambda v:v[0])
        improve=base-cost
        keep=improve>=max(cfg['identity_min_absolute_improvement'],cfg['identity_min_relative_improvement']*base)
        if not keep: delta=np.zeros(3);cost=base
        diagnostic={'status':'optimized' if keep else 'identity_geometric_gate',
                    'success':True,'identity_fallback':False,'converged':bool(converged),
                    'initial_objective':float(base),'selected_objective':float(cost),
                    'improvement':float(improve),'focal_observations':nf,'current_actors':na,
                    'matched_observations':int(assoc[-1].sum()),'valid_segments':len(segment_a),
                    'selected_inner_evaluations':evaluations,'elapsed_s':time.perf_counter()-start_time,
                    'at_bound':bool((np.abs(delta)>=.99*bound).any())}
        result=delta.copy();result[2]=np.deg2rad(result[2])
        return result,diagnostic
