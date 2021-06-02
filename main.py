#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spherical sampling of rotational degrees of freedom
"""
import argparse
import os
import numpy as np
from ape.sampling import SamplingJob
from ape.qchem import QChemLog
from tnuts.qchem import get_level_of_theory
from tnuts.main import MCMCTorsions, NUTS_run
from ads.sph_harm import Yij as Y
from ads.adsorbate import Adsorbate
from ads.graphics import make_fig

def parse_command_line_arguments(command_line_args=None):
    
    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a frequency file describing the job to execute')
    parser.add_argument('-n', type=int, help='number of CPUs to run quantum calculation')
    parser.add_argument('-nads', type=str, help='number of atoms in the adsorbate')
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')
    parser.add_argument('-ncirc', type=int, help='number of circles')
    parser.add_argument('-hpc', type=bool, help='if run on cluster')

    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]
    return args

def main():
    # IF JOB HAS ALREADY COMPLETED, WE JUST WANT TO ANALYZE THE DATA FROM .npy FILES
    args = parse_command_line_arguments()
    input_file = args.file.split('/')[-1]
    project_directory = os.path.abspath(os.path.dirname(args.file))
    path = os.path.join(project_directory, '{}.npy')
    ncpus = int(args.n) if args.n is not None else 4
    T = float(args.T) if args.T is not None else 300
    ncirc = int(args.ncirc) if args.ncirc is not None else 25
    hpc = bool(args.hpc)
    nads = int(args.nads)
    print("Number of adsorbates is", nads)
    ads = Adsorbate(input_file, project_directory, nads, ncpus=ncpus)
    if not os.path.exists(path.format('xsph')) and not os.path.exists(path.format('v')):
        if not T:
            T = 300
        x,xsph,v = ads.sample(na=ncirc, write=True)

        np.save(path.format('xcart'), x)
        np.save(path.format('xsph'), xsph)   
        np.save(path.format('v'), v)

    xsph = np.load(path.format('xsph'))
    v = np.load(path.format('v'))

    Ymat = Y(xsph, lmax=30)
    
    # SOLVE THE COEFFICIENTS
    U,S,V = np.linalg.svd(Ymat, full_matrices=False)
    ahat = np.matmul( np.linalg.pinv(np.matmul(U, np.matmul( np.diag(S), V))), np.array(v))
    
    print(ahat)
    inert = ads.get_moment_of_inertia()[0]
    #print(inert.get_partition_function(300))
    #print(inert.get_entropy(300))

    print(np.power(np.array(inert.modes[1].inertia.value), -1.))
    I = np.sum(np.power(np.array(inert.modes[1].inertia.value), -1.))
    print(I)
    print(inert.modes[1].inertia.value)

    #make_fig(xsph, v, project_directory, lmax=30)


if __name__ == '__main__':
    main()
