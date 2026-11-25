"""Observation weighting and fixed logarithmic fusion."""

import numpy as np

from .geometry import wrap
from .se2 import log_se2, exp_se2


def observations(p, h, mask):
    times=np.arange(4,50,5)
    vel=p[:,times]-p[:,times-4]
    travel=np.linalg.norm(vel,axis=-1)
    direction=np.arctan2(vel[...,1],vel[...,0])
    angle=h[:,times]
    weights=(~mask[:,times])*(~mask[:,times-4])*(~mask[:,-1,None])
    weights=weights*np.minimum(travel/.8,1.)*np.maximum(np.cos(wrap(direction-angle)),0.)**2
    weights/=np.maximum(weights.sum(axis=1,keepdims=True),1.)
    good=weights>1e-6
    if not good.any():
        return p.reshape(-1,2)[:0],h.reshape(-1)[:0],np.empty(0),np.empty(0,int),np.empty(0,int)
    ii,jj=np.where(good)
    return p[ii,times[jj]],angle[good],weights[good]/weights.sum(),ii,times[jj]

def fixed_prediction(proposal, initial, weights):
    vectors=log_se2(np.array([initial,proposal,np.zeros(3)]))
    return exp_se2(np.r_[np.sum(weights[0,:,None]*vectors[:,:2],axis=0),weights[1]@vectors[:,2]])
