"""Lie algebra operations used by HMCA."""

import numpy as np


def jacobian_coefficients(yaw):
    a=np.sinc(yaw/np.pi)
    # (1 - cos(yaw))/yaw, evaluated without cancellation near zero.
    b=.5*yaw*np.sinc(yaw/(2*np.pi))**2
    return a,b

def log_se2(delta):
    a,b=jacobian_coefficients(delta[...,2])
    det=a*a+b*b
    x,y=delta[...,0],delta[...,1]
    return np.stack([(a*x+b*y)/det,(a*y-b*x)/det,delta[...,2]],axis=-1)

def exp_se2(xi):
    a,b=jacobian_coefficients(xi[...,2])
    x,y=xi[...,0],xi[...,1]
    return np.stack([a*x-b*y,b*x+a*y,xi[...,2]],axis=-1)
