# -*- coding: utf-8 -*-
import numpy as np
from math import sqrt
from math import factorial as fact
import rmgpy.constants as constants
from ape.HarmonicBasis import IntXHmHnexp
from ape.FourierBasis import IntXPhimPhin
from sympy.physics.wigner import gaunt

hbar1 = constants.hbar / constants.E_h # in hartree*s
hbar2 = constants.hbar * 10 ** 20 / constants.amu # in amu*angstrom^2/s

def Hlmllmm(ahat, I, l, m, ll, mm):
    result = 0
    # Loop over fitting coefficients
    for L in range(int(np.sqrt(len(ahat)))):
        for M in np.linspace(-L, L, 2*L+1):
            k = int(np.power(L,2) + L + M)
            #multiplier = 1
            #if mm != 0:
            #    multiplier *= np.sqrt(2)
            #if m != 0:
            #    multiplier *= np.sqrt(2)
            #if M != 0:
            #    multiplier *= np.sqrt(2)
            ## â should carry the unit of energy
            #result += ahat[k] * multiplier*\
            #        float(gaunt(ll,L,l,mm,int(M),m, prec=64))
            result += ahat[k] * float(gaunt(ll,L,l,mm,int(M),m, prec=16))

    if ll == l and mm == m:
        result += hbar1*hbar2* l*(l+1)/2 / I
    return result # in Hartree

def set_anharmonic_H(ahat, I, Lam, Lam_prev, H_prev):
    size = Lam**2
    H = np.zeros((size, size), np.float64)
    for l in range(Lam):
        for m in np.linspace(-l, l, 2*l+1):
            j = int(np.power(l,2)+l+m)

            for ll in range(l+1):
                for mm in np.linspace(-ll, ll, 2*ll+1):
                    i = int(np.power(ll,2)+ll+mm)

                    if l < Lam_prev and ll < Lam_prev:
                        print("Calculating from previous")
                        hval = H_prev[i][j]
                    else: # l previously not calculated:
                        hval = Hlmllmm(ahat, I, l, int(m), ll, int(mm))
                    #print(hval)
                    H[i][j] = hval
                    H[j][i] = hval
    return H

def H(ahat, I, Lam, H_prev):
    lam = int(np.sqrt(len(ahat)))
    Lam = 12
    M = Lam**2
    H = np.zeros(shape=(M,M))
    #ahat = np.zeros(shape=(lam**2))
    hbar = 1 # FOR NOW
    I = 1 # FOR NOW should be in amu.angstrom^2/s
    for l in range(Lam):
        for m in np.linspace(-l,l,2*l+1):
            j = int(np.power(l, 2) + l + m)
            for ll in range(Lam):
                for mm in np.linspace(-ll,ll,2*ll+1):
                    i = int(np.power(ll, 2) + ll + mm)
                    if j > i:
                        H[i][j] = H[j][i]
                        continue
                    for L in range(lam):
                        for M in np.linspace(-L, L, 2*L+1):
                            k = int(np.power(L,2) + L + M)
                            multiplier = 1
                            if mm != 0:
                                multiplier *= np.sqrt(2)
                            if m != 0:
                                multiplier *= np.sqrt(2)
                            if M != 0:
                                multiplier *= np.sqrt(2)
                            # â should carry the unit of energy
                            H[i][j] += ahat[k] * multiplier* float(gaunt(ll,L,l,int(mm),int(M),int(m), prec=6))
                            #H[i][j] += ahat[k] * real_gaunt(ll, L, l, int(mm), -int(M), int(m))
                    if ll == l and mm == m:
                        # should have units of energy consistent with â 
                        H[i][j] += hbar2 * l*(l+1)/2. * np.power(I, -1.)


def Hmn(m, n, ahat, xsph, v):
    result = 0
    if is_tors:
        # use fourier basis function
        I = mode_dict[mode]['M'] # in amu*angstrom^2
        k = mode_dict[mode]['K'] # 1/s^2
        step_size = mode_dict[mode]['step_size']
        delta_q = sqrt(I) * step_size # in sqrt(amu)*angstrom
        L = np.pi * sqrt(I) # in sqrt(amu)*angstrom
        x2 = 0
        for i in sorted(polynomial_dict[mode].keys()):
            x1 = x2
            x2 += delta_q
            a = [polynomial_dict[mode][i][ind] for ind in ['ai','bi','ci','di']]
            result += IntXPhimPhin(m,n,x1,x2,L,a)
        if (m==n and m!=0):
            if (m%2==0): m /= 2
            else: m = (m+1)/2
            result += pow(m*np.pi/L,2)*(hbar1*hbar2)/2 # in hartree
        
def SetAnharmonicH(ahat, xsph, v, size, N_prev, H_prev):
    H = np.zeros((size, size), np.float64)
    for m in range(size):
        for n in range(m+1):
            if m < N_prev and n < N_prev:
                Hmn_val = H_prev[m][n]
            else:
                Hmn_val = Hmn(m, n, polynomial_dict, mode_dict, energy_dict, mode, is_tors)
            H[m][n] = Hmn_val
            H[n][m] = Hmn_val
    return H
