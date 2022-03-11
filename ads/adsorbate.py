#/usr/bin env

# import needed modules
import os
import sys
from ape.sampling import SamplingJob
from ape.qchem import QChemLog
from ape.common import get_electronic_energy
from ape.InternalCoordinates import getXYZ
from tnuts.qchem import get_level_of_theory, load_adsorbate
from tnuts.job.job import Job
from rigidObject import RigidObject
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import numgrid
from ads.hamiltonian import set_anharmonic_H
from sympy.physics.wigner import gaunt

class Adsorbate:
    """
    This class is where all of the adsorbate orientational sampling is done.
    The geometries are obtained from a standard Q-Chem frequency file. Helper
    functions to parse the freq-file outputs are obtained from the APE and
    TNUTS modules. The Numgrid package discretizes the SO(3) domain into a
    Lebedev grid. Rotations are performed through the rigidObject class, which
    is created from adsorbate coordinates.
    """
    def __init__(self, freqfile, directory, nads, T=300, P=101325, ncpus=4,
                grid='healpix'):
        self.freqfile = freqfile
        self.directory = directory
        self.nads = nads
        self.T = T
        self.result_info = list()
        self.P = P
        print(self.nads, type(self.nads))

        self.grid = 'healpix'

        self.label = freqfile.split('.')[0]
        self.Log = QChemLog(os.path.join(directory, freqfile))
        level_of_theory_kwargs = get_level_of_theory(self.Log)

        self.samp = SamplingJob(
                input_file=os.path.join(directory, freqfile),
                label=self.label,
                ncpus=ncpus, output_directory=directory,
                thresh=0.5,
                **level_of_theory_kwargs)
        self.samp.parse()

        self.path = os.path.join(self.samp.output_directory, 'output_file')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_name = 'samp_{}'

        # get qchem kwargs
        # these are all that are needed with APE, with the exception of 'path' and 'file_name'
        self.qchem_kwargs = {}
        self.qchem_kwargs['ncpus'] = self.samp.ncpus
        self.qchem_kwargs['charge'] = self.samp.charge
        self.qchem_kwargs['level_of_theory'] = self.samp.level_of_theory
        self.qchem_kwargs['basis'] = self.samp.basis
        self.qchem_kwargs['unrestricted'] = self.samp.unrestricted
        self.qchem_kwargs['multiplicity'] = self.samp.spin_multiplicity
        # kwargs for just adsorbate alone in gas phase
        self.ads_kwargs = {}
        self.ads_kwargs['ncpus'] = self.samp.ncpus
        self.ads_kwargs['charge'] = self.samp.charge
        self.ads_kwargs['level_of_theory'] = self.samp.level_of_theory
        self.ads_kwargs['basis'] = self.samp.basis
        self.ads_kwargs['unrestricted'] = self.samp.unrestricted
        self.ads_kwargs['multiplicity'] = self.samp.spin_multiplicity 

        # QM kwargs
        self.qchem_kwargs['is_QM_MM_INTERFACE'] = self.samp.is_QM_MM_INTERFACE
        self.qchem_kwargs['QM_USER_CONNECT'] = self.samp.QM_USER_CONNECT
        self.qchem_kwargs['QM_ATOMS'] = self.samp.QM_ATOMS
        self.qchem_kwargs['force_field_params'] = self.samp.force_field_params
        self.qchem_kwargs['fixed_molecule_string'] = self.samp.fixed_molecule_string
        self.qchem_kwargs['opt'] = self.samp.opt
        self.qchem_kwargs['number_of_fixed_atoms'] = self.samp.number_of_fixed_atoms

        # get mass of adsorbate
        print("number of adsorbates", self.nads)
        #mass = self.samp.conformer.mass.value_si[:self.nads]
        #coordinates = self.samp.conformer.coordinates.value[:self.nads]

        coordinates, adsnums, mass = load_adsorbate(self.Log, nads)
        self.mycoords = self.Log.load_geometry()[0]

        #if linear is None:
        #    linear = is_linear(coordinates)
        #    if linear:
        #        logging.info('Determined species {0} to be linear.'.format(label))

        # get center of mass of adsorbate
        # THIS WILL STAY CONSTANT
        xm = 0.0
        ym = 0.0
        zm = 0.0
        totmass = 0.0
        for i in range(self.nads):
            xm += mass[i] * coordinates[i, 0]
            ym += mass[i] * coordinates[i, 1]
            zm += mass[i] * coordinates[i, 2]
            totmass += mass[i]
        xm /= totmass
        ym /= totmass
        zm /= totmass

        print("CENTER OF MASS IS AT", (xm, ym, zm))
        print("COORDS ARE", self.samp.internal.c3d)
        self.com = np.array([xm,ym,zm])    # AGAIN, THIS WILL STAY CONSTANT
        self.com = np.array([0,0,0])
        print("NEW CENTER OF MASS IS AT", (0,0,0))
        print("NEW COORDS ARE", self.mycoords)
        self.rigidrotor = RigidObject(nads, self.mycoords)
        self.rigidrotor.change_origin(self.com) # Set rotational origin to c.o.m.
        self.record_script ='''{natom}\n# Point {sample} Energy = {e_elect}\n{xyz}\n'''

    def reorient_by(self, dtheta, dphi):
        # theta is polar angle, phi is azimuthal angle
        self.rigidrotor.rotate_y(dtheta)
        self.rigidrotor.rotate_z(dphi)
        self.samp.internal.cart_coords[:(3*self.nads)] = self.rigidrotor.to_array().flatten()
        self.mycoords[:self.nads] = self.rigidrotor.to_array()

    def sample(self, na=20, dry=False, write=False, which_grid='lebedev'):
        # Returns a cartesian and spherical grid.
        # na is a relic of past implementation and is not used.
        # spherical grid order = (θ, φ)

        # filename for vmd viewing
        name = 'configs.txt'

        # begin sampling spherical grid
        sph_grid = []
        v = []
        count = 0

        # Strongly recommended to not use 'healpix' so comments in this block
        # of code are very limited.
        if which_grid == 'healpix':
            # CONSTRUCT HEALPix grid of Nside = 2
            # Needs healpy package
            import healpy as hp
            nside = 2
            theta, phi = hp.pix2ang(nside, np.arange(0,hp.nside2npix(nside)))
            xyz = hp.ang2vec(theta,phi)

            # Set equilibrium (initial) theta, phi, chi to (0.,0.,0.)
            thprev = 0.
            phprev = 0.
            chprev = 0.
            chi = np.linspace(0, 2*np.pi, int(np.floor(2*np.pi/hp.nside2resol(nside))+1))[:-1]
            count = 0   # used to label job numbers
            grid = hp.ang2vec(theta,phi)
            for th,ph in zip(theta,phi):
                self.reorient_by(0, ph-phprev)
                self.reorient_by(th-thprev, 0)
                for ch in chi:
                    vec = hp.ang2vec(th,ph)
                    self.rigidrotor.rotate(ch-chprev, np.cos(ph)*np.sin(th),
                            np.sin(ph)*np.sin(th), np.cos(th))
                    self.mycoords[:self.nads] = self.rigidrotor.to_array()

                    xyz = getXYZ(self.samp.symbols, self.mycoords.ravel())
                    if not dry:
                        E = get_electronic_energy(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
                                    **self.qchem_kwargs)
                        v.append(E)
                    else:
                        E = 0
                        make_job(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
                                **self.qchem_kwargs)
                        if write:
                            # write a file showing geometric configurations to
                            # be sampled
                            name = 'configs.txt'
                            with open(os.path.join(self.directory, name), 'a') as f:
                                content = self.record_script.format(natom=self.samp.natom,
                                        sample=count, e_elect=E, xyz=xyz)
                                f.write(content)

                    sph_grid.append(np.array([th,ph,ch]))
                    chprev = ch
                    thprev = th
                    phprev = ph
                    count += 1

        # Strongly recommended to use Lebedev grids!
        if which_grid == 'lebedev':
            # CONSTRUCT LEBEDEV GRID
            npoints = 50    # choices {6, 14, 26, 38, 50, 74, 86, 110, 146,
                            #           170, ...}
            # Set equilibrium (initial) theta, phi, chi to (0.,0.,0.)
            thprev = 0.
            phprev = 0.
            chprev = 0.

            # Obtain Lebedev grid in (θ,φ)
            # (χ is obtained by uniform discretization).
            xyz,wts = numgrid.angular_grid(npoints)
            resol = 2*np.sqrt(np.pi/npoints)
            chi = np.linspace(0,2*np.pi, int(np.floor(2*np.pi/resol)))[:-1] # χ
            XYZ = np.array([np.array(coord) for coord in xyz])
            X,Y,Z = XYZ[:,0], XYZ[:,1], XYZ[:,2]
            theta,phi = np.arctan2(np.sqrt(np.power(X,2)+np.power(Y,2)),Z),\
                            np.arctan2(Y,X) # θ, φ
            angs = np.array(list(sorted(zip(theta,phi%(2*np.pi)),
                key=lambda t: t[0]))) # angle combinations for spherical grid sorted by theta

            count = 0   # to label job numbers
            for th,ph in angs:
                # Reorient adsorbate in increments of Δθ, Δφ.
                self.reorient_by(0, ph-phprev)
                self.reorient_by(th-thprev, 0)
                # For each orientation of (θ,φ), sample a full χ rotation
                # (around molecular z axis)
                for ch in chi:
                    print("Shifting by:", np.around(th-thprev,3), np.around(ph-phprev,3), np.around(ch-chprev,3),
                          '\t', "Now at", np.around(th,3), np.around(ph,3), np.around(ch,3))
                    # molecular z rotation is done around vector (x,y,z) specified by
                    # *vec = (θ, φ)
                    vec = hp.ang2vec(th,ph) # not used
                    # Do the χ rotation around *vec
                    self.rigidrotor.rotate(ch-chprev, np.cos(ph)*np.sin(th),
                            np.sin(ph)*np.sin(th), np.cos(th))
                    # Set adsorbate geometry to rotated geometry
                    self.mycoords[:self.nads] = self.rigidrotor.to_array()
                    xyz = getXYZ(self.samp.symbols, self.mycoords.ravel())

                    if not dry:
                        # do the calculation using helper function
                        E = get_electronic_energy(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
                                    **self.qchem_kwargs)
                        v.append(E)
                    else:
                        # or just write the inputs with a dummy E used for VMD
                        # software, also using helper function
                        E = 0
                        make_job(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
                                **self.qchem_kwargs)
                        if write:
                            # write a file showing geometric configurations to sample
                            name = 'configs.txt'
                            with open(os.path.join(self.directory, name), 'a') as f:
                                content = self.record_script.format(natom=self.samp.natom,
                                        sample=count, e_elect=E, xyz=xyz)
                                f.write(content)

                    # update variables for next iteration
                    sph_grid.append(np.array([th,ph,ch]))
                    chprev = ch
                    thprev = th
                    phprev = ph
                    count += 1

        if not dry:
            # set 0 of energy to the minimum energy sampled (should be
            # equilibrium energy at (0.,0.,0.) )
            v = np.array(v)-np.min(v)
        # Return θ,φ,χ grid and energy grid
        return np.array(sph_grid), np.array(v)

    def solv_eig(self, ahat, T):
        """
        Not implemented, this was done by solving ahat coefficients, then using
        a C++ executable to diagonalize rotational Hamiltonian.
        """
        return

    def calc_thermo(self, eig, Lam, T):
        """
        Not implemented, this was done by solving ahat coefficients, then using
        a C++ executable to diagonalize rotational Hamiltonian.
        """
        return

    def get_symmetry_number(self):
        """
        Rotational symmetry number of adsorbate
        """
        Log = QChemLog(os.path.join(self.directory,
            'ads{}.q.out'.format(self.nads)))
        symmetry = Log.load_conformer[0].modes[1].symmetry
        return symmetry

    def get_moment_of_inertia(self):
        """
        Rotational moments of inertia or contained within conformer object.
        """
        if not os.path.exists(os.path.join(self.directory,
            'ads{}.q.out'.format(self.nads))):
            ads_xyz = getXYZ(self.samp.symbols[:self.nads],
                    self.samp.internal.cart_coords[:(3*self.nads)])
            adsJob = Job(xyz=ads_xyz, path=self.directory,
                    file_name='ads{}'.format(self.nads), jobtype='freq', **self.ads_kwargs)
            adsJob.write_input_file()
            adsJob.submit()
        Log = QChemLog(os.path.join(self.directory,
            'ads{}.q.out'.format(self.nads)))
        return Log.load_conformer()


def make_job(xyz, path, file_name, ncpus, charge=None, multiplicity=None, level_of_theory=None, basis=None, unrestricted=None, \
        is_QM_MM_INTERFACE=None, QM_USER_CONNECT=None, QM_ATOMS=None, force_field_params=None, fixed_molecule_string=None, opt=None, number_of_fixed_atoms=None):
    """
    This uses helper functions from APE to create QM/MM jobs with the same
    $rem$ variables as the Q-Chem freq-file.
    """
    #file_name = 'output'
    if is_QM_MM_INTERFACE:
        # Create geometry format of QM/MM system 
        # <Atom> <X> <Y> <Z> <MM atom type> <Bond 1> <Bond 2> <Bond 3> <Bond 4>
        # For example:
        # O 7.256000 1.298000 9.826000 -1  185  186  0 0
        # O 6.404000 1.114000 12.310000 -1  186  713  0 0
        # O 4.077000 1.069000 0.082000 -1  188  187  0 0
        # H 1.825000 1.405000 12.197000 -3  714  0  0 0
        # H 2.151000 1.129000 9.563000 -3  189  0  0 0
        # -----------------------------------
        QMMM_xyz_string = ''
        for i, xyz in enumerate(xyz.split('\n')):
            QMMM_xyz_string += " ".join([xyz, QM_USER_CONNECT[i]]) + '\n'
            if i == len(QM_ATOMS)-1:
                break
        QMMM_xyz_string += fixed_molecule_string
        job = Job(QMMM_xyz_string, path, file_name,jobtype='sp', ncpus=ncpus, charge=charge, multiplicity=multiplicity, \
                level_of_theory=level_of_theory, basis=basis, unrestricted=unrestricted, QM_atoms=QM_ATOMS, \
                force_field_params=force_field_params, opt=opt, number_of_fixed_atoms=number_of_fixed_atoms)
    else:
        job = Job(xyz, path, file_name,jobtype='sp', ncpus=ncpus, charge=charge, multiplicity=multiplicity, \
                level_of_theory=level_of_theory, basis=basis, unrestricted=unrestricted)

        # Write Q-Chem input file
    job.write_input_file()
