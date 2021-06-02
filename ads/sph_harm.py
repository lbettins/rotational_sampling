# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from math import pi, sin, cos
import scipy.special

# WIKIPEDIA BASIS
def Ybasis_wiki(l,m,theta,phi):
    if m < 0:
        return np.sqrt(2)*np.imag(scipy.special.sph_harm(np.abs(m),l,phi,theta))
    elif m == 0:
        return np.real(scipy.special.sph_harm(0,l,phi,theta))
    else:
        return np.sqrt(2)*np.real(scipy.special.sph_harm(m,l,phi,theta))

# WIKIPEDIA BASIS
def Yij_wiki(gridpts,lmax):
    # initialize basis function fitting matrix
    #
    # inputs:
    #    - gridpts -- an array of all (θ,φ) pairs (k x 2)
    #                 order is important! should be given as θ first, then φ
    #    - lmax -- max spherical harmonic (should be even)
    #
    # outputs:
    #    - Yij -- the regression basis matrix
    #
    # rows x cols  <-->  k x M
    # i = [0, k-1]  where k = # of gridpoints
    # j = [0, M] where M = # of basis functions, (lmax+1)^2
    #
    k = len(gridpts)
    #y = scipy.special.sph_harm
    Yij = np.zeros(shape=(k,(lmax+1)**2))
    for i in range(k):
        j = 0
        for l in range(lmax):
            for m in np.linspace(-l,l,2*l+1):
                Yij[i][int(l**2+l+m)] += np.sqrt(2)*np.imag(scipy.special.sph_harm(np.abs(m),l,gridpts[i][1], gridpts[i][0])) if m < 0 \
                    else np.real(scipy.special.sph_harm(0,l,gridpts[i][1], gridpts[i][0])) if m == 0 \
                    else np.sqrt(2)*np.real(scipy.special.sph_harm(m,l,gridpts[i][1], gridpts[i][0]))
                j += 1
    return Yij

### DESCOTEAUX BASIS ###
def Ybasis(l,m,theta,phi):
    if m < 0:
        return np.sqrt(2)*np.real(scipy.special.sph_harm(m,l,phi,theta))
    elif m == 0:
        return np.real(scipy.special.sph_harm(0,l,phi,theta))
    else:
        return np.sqrt(2)*np.imag(scipy.special.sph_harm(m,l,phi,theta))
    
# DESCOTEAUX BASIS
def Yij(gridpts,lmax):
    # initialize basis function fitting matrix
    #
    # inputs:
    #    - gridpts -- an array of all (θ,φ) pairs (k x 2)
    #                 order is important! should be given as θ first, then φ
    #    - lmax -- max spherical harmonic (should be even)
    #
    # outputs:
    #    - Yij -- the regression basis matrix
    # 
    # rows x cols  <-->  k x M
    # i = [0, k-1]  where k = # of gridpoints
    # j = [0, M] where M = # of basis functions, (lmax+1)^2
    #
    k = len(gridpts)
    #y = scipy.special.sph_harm
    Yij = np.zeros(shape=(k,(lmax+1)**2))
    for i in range(k):
        j = 0
        for l in range(lmax):
            for m in np.linspace(-l,l,2*l+1):
                Yij[i][int(l**2+l+m)] += np.sqrt(2)*np.real(scipy.special.sph_harm(m,l,gridpts[i][1], gridpts[i][0])) if m < 0 \
                    else np.real(scipy.special.sph_harm(0,l,gridpts[i][1], gridpts[i][0])) if m == 0 \
                    else np.sqrt(2)*np.imag(scipy.special.sph_harm(m,l,gridpts[i][1], gridpts[i][0]))
                j += 1
    return Yij
