#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spherical grid sampling of rotational degrees of freedom
"""
import argparse
import os
import numpy as np
from ape.sampling import SamplingJob
from ape.qchem import QChemLog
from ads.adsorbate import Adsorbate
#from tnuts.qchem import get_level_of_theory
#from tnuts.main import MCMCTorsions, NUTS_run
#from ads.sph_harm import Yij as Y
#from ads.graphics import make_fig

def parse_command_line_arguments(command_line_args=None):
    # Arguments to include when executing code
    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='Q-Chem frequency file describing the job to execute')
    parser.add_argument('-n', type=int, help='number of CPUs to run quantum calculation')
    parser.add_argument('-nads', type=str, help='number of atoms in the adsorbate')
    parser.add_argument('-g', type=str, help='gridtype (use "lebedev")')
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')
    parser.add_argument('-ncirc', type=int, help='number of gridpoints')
    parser.add_argument('-hpc', type=bool, help='if run on cluster')
    parser.add_argument('-dry', type=bool, help='just write scripts')
    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]
    return args

def main():
    # If job has already completed, we just want to analyze the data from .npy
    # files
    # First get command line arguments:
    args = parse_command_line_arguments()
    input_file = args.file.split('/')[-1]
    project_directory = os.path.abspath(os.path.dirname(args.file))
    path = os.path.join(project_directory, '{}.npy')
    ncpus = int(args.n) if args.n is not None else 4    # no. threads
    T = float(args.T) if args.T is not None else 298    # stand. temp.
    # ncirc is no longer used / is a relic of past implementation
    ncirc = int(args.ncirc) if args.ncirc is not None else 25
    hpc = bool(args.hpc)
    nads = int(args.nads)
    dry = bool(args.dry)
    if args.g is not None:
        gridtype = str(args.g)
    else:
        gridtype = 'healpix'    # no clear quadrature rules (avoid)

    print("Number of adsorbates is", nads)
    print("Sampling will be done with {} grid.".format(gridtype))
    # Create Adsorbate object to facilitate sampling
    ads = Adsorbate(input_file, project_directory, nads, ncpus=ncpus)

    # If sampling has been done already, a pickle will exist named 'xsph.npy', whose
    # energies are in 'v.npy', and we can skip this part
    if not os.path.exists(path.format('xsph')) and not os.path.exists(path.format('v')):
        if not T:
            T = 298
        if dry: # just write the inputs
            x,xsph,v = ads.sample(na=ncirc, dry=dry, write=True, which_grid=gridtype)
            return
        else:
            xsph,v = ads.sample(na=ncirc, dry=dry, write=True, which_grid=gridtype)
            np.save(path.format('xsph'), xsph)
            np.save(path.format('v'), v)

    # Load 'xsph.npy' and 'v.npy' arrays
    xsph = np.load(path.format('xsph'))
    v = np.load(path.format('v'))

    # This is where coefficient fitting in the Wigner D-Mat. elem. basis is
    # done. This currently exists in a jupyter notebook.


if __name__ == '__main__':
    main()
