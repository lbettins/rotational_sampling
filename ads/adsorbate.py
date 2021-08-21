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
import healpy as hp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import scipy.special
from ads.hamiltonian import set_anharmonic_H
from sympy.physics.wigner import gaunt

class Adsorbate:
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

    def sample(self, na=20, dry=False, write=False, grid='healpix'):
        # Returns a cartesian and spherical grid
        # na is the sampling resolution
        # spherical grid order = (θ, φ)
        # na = number of circular slices

        # filename for vmd viewing
        name = 'configs.txt'

        # begin sampling spherical grid
        grid = []
        sph_grid = []
        v = []
        count = 0

        #dc = np.pi/(na-1); # intrinsic rotation (spin) step

        #da = np.pi/(na-1); # latitude angle step
        #a = np.pi/2
        #actual_a = 0.
        print('THETA', 'PHI', 'CHI', 'dTHETA', 'dPHI', 'dCHI')

        if grid == 'healpix':
            # CONSTRUCT HEALPix grid of Nside = 2 (user-changed)
            nside = 2
            theta, phi = hp.pix2ang(nside, np.arange(0,hp.nside2npix(nside)))
            xyz = hp.ang2vec(theta,phi)

            thprev = 0.
            phprev = 0.
            chprev = 0.
            chi = np.linspace(0, 2*np.pi, int(np.floor(2*np.pi/hp.nside2resol(nside))+1))[:-1]
            count = 0
            grid = hp.ang2vec(theta,phi)
            for th,ph in zip(theta,phi):
                self.reorient_by(0, ph-phprev)
                self.reorient_by(th-thprev, 0)
                for ch in chi:
                    print("Shifting by:", np.around(th-thprev,3), np.around(ph-phprev,3), np.around(ch-chprev,3),
                          '\t', "Now at", np.around(th,3), np.around(ph,3), np.around(ch,3))
                    vec = hp.ang2vec(th,ph)
                    print("Z axis vec", *vec)
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
                            # write a file showing geometric configurations to sample
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
        if grid == 'lebedev':
            # CONSTRUCT LEBEDEV GRID
            npoints = 50    # choices {6, 14, 26, 38, 50, 74, 86, 110, 146,
                            #           170, ...}
            thprev = 0.
            phprev = 0.
            chprev = 0.
            xyz,wts = numgrid.angular_grid(npoints)
            resol = 2*np.sqrt(np.pi/npoints)
            chi = np.linspace(0,2*np.pi, int(np.floor(2*np.pi/resol)))[:-1]
            XYZ = np.array([np.array(coord) for coord in xyz])
            X,Y,Z = XYZ[:,0], XYZ[:,1], XYZ[:,2]
            theta,phi = np.arctan2(np.sqrt(np.power(X,2)+np.power(Y,2)),Z),\
                            np.arctan2(Y,X)
            angs = np.array(list(sorted(zip(theta,phi%(2*np.pi)),
                key=lambda t: t[0]))) # angle combinations for spherical grid sorted by theta
            count = 0
            for th,ph in angs:
                self.reorient_by(0, ph-phprev)
                self.reorient_by(th-thprev, 0)
                for ch in chi:
                    print("Shifting by:", np.around(th-thprev,3), np.around(ph-phprev,3), np.around(ch-chprev,3),
                          '\t', "Now at", np.around(th,3), np.around(ph,3), np.around(ch,3))
                    vec = hp.ang2vec(th,ph)
                    print("Z axis vec", *vec)
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
                            # write a file showing geometric configurations to sample
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

        #for ia in range(na): # slice sphere to circles in xy planes
        #    r=np.cos(a);                           # radius of actual circle in xy plane
        #    z=np.sin(a);                           # height of actual circle in xy plane
        #    nb=np.ceil(2.0*np.pi*r/da)
        #    db=2.0*np.pi/nb;             # longitude angle step
        #    if ia==0 or ia==na-1:
        #        nb=1
        #        db=0.0 # handle edge cases

        #    b = 0.0
        #    for ib in range(int(nb)):  # cut circle to vertexes
        #        x=r*np.cos(b);                     # compute x,y of vertex
        #        y=r*np.sin(b);
        #        grid.append(np.array([x,y,z]))
        #        sph_grid.append(np.array([actual_a,b]))

        #        nc = np.ceil(2.0*np.pi/dc)
        #        c = 0.0
        #        for ic in range(int(nc)):
        #            # get XYZ for = 0 first!
        #            xyz = getXYZ(self.samp.symbols, self.samp.internal.cart_coords)
        #            xyz = getXYZ(self.samp.symbols, self.mycoords.ravel())

        #            print(actual_a,b,c,'\t',da,db,dc)
        #            # run Q-Chem job
        #            if not dry:
        #                E = get_electronic_energy(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
        #                        **self.qchem_kwargs)
        #                v.append(E)
        #            else:
        #                E = 0
        #                make_job(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
        #                        **self.qchem_kwargs)
        #                if write:
        #                    # write a file showing geometric configurations to sample
        #                name = 'configs.txt'
        #                with open(os.path.join(self.directory, name), 'a') as f:
        #                    content = self.record_script.format(natom=self.samp.natom,
        #                            sample=count, e_elect=E, xyz=xyz)
        #                    f.write(content)
        #            if nb == 1:
        #                count += 1
        #                break

        #            c += dc
        #            self.rigidrotor.rotate(dc,x,y,z)
        #            self.mycoords[:self.nads] = self.rigidrotor.to_array()
        #            count += 1

        #        self.reorient_by(0,db) #(dtheta, dphi)
        #        b += db

        #    a -= da
        #    actual_a += da
        #    self.reorient_by(da,0)  #(dtheta, dphi)

        if not dry:
            v = np.array(v)-np.min(v)
            #np.save
        return np.array(grid), np.array(sph_grid), np.array(v)

    def solv_eig(self, ahat, T):
        # Solve eigenvalues for ROTATION only
        Lam = 18
        Lam_prev = 0
        H_prev = None
        Qold = np.log(sys.float_info[0])
        converge = False
        while not converge:
            Lam += 1
            H = set_anharmonic_H(ahat, I,
                    Lam, Lam_prev,
                    H_prev=H_prev)
            H_prev = deepcopy(H)
            eig, v = np.linalg.eigh(H)
            E, S, F, Q, Cv = self.calc_thermo(eig, Lam, T)

            if Qold == np.log(sys.float_info[0]):
                self.result_info.append("# \n# \t %d \t\t-\t\t-" % Nbasis) #first run
            else:
                self.result_info.append("# \n# \t %d \t\t %.10f \t\t %.10f" % (Nbasis,abs(Q-Qold)))

            #if ((abs(Q-Qold)<1e-4)):
            if ((abs(Q-Qold)<100)):
                self.result_info.append("# Convergence criterion met")
                self.result_info.append("# ------------------------------------")
                converge = True
                self.result_info.append("# Frequency (cm-1): %.10f" % v)
                self.result_info.append("# Zero point vibrational energy (hartree): %.10f" % E0)
                self.result_info.append("# Energy (hartree): %.10f" % E )
                self.result_info.append("# Entropy (hartree/K): %.10f" % S)
                self.result_info.append("# Free energy (hartree): %.10f" % F)
                self.result_info.append("# Partition function: %.10f" % Q)
                hartree2kcalmol = constants.E_h * constants.Na / 4184
                E0 *= hartree2kcalmol
                E *= hartree2kcalmol
                S *= hartree2kcalmol * 1000
                F *= hartree2kcalmol
                Cv *= hartree2kcalmol * 1000
                '''
                print("Frequency (cm-1): ",v)
                print("Zero point vibrational energy (kcal/mol): ",E0)
                print("Energy (kcal/mol): ",E )
                print("Entropy (cal/mol/K): ",S)
                print("Free energy (kcal/mol): ",F)
                print("Partition function: ",Q)
                '''

            Qold = Q
        print("Converged Lambda =", Lam)
        return E, S, F, Q, Cv

    def calc_thermo(self, eig, Lam, T):
        N = Lam**2
        beta = 1/(constants.kB*T) * constants.E_h

        Q = 0
        E = 0
        dQ = 0
        ddQ = 0
        for i in range(N):
            Ei = eig[i]
            Q += exp(-beta*Ei)
            dQ += Ei*exp(-beta*Ei)*beta/T
            ddQ += -2*Ei*exp(-beta*Ei)*beta/pow(T,2) + pow(Ei,2)*exp(-beta*Ei)*pow(beta,2)/pow(T,2)
            E += Ei*exp(-beta*Ei)
        E /= Q
        if True:
            omega = self.get_symmetry_number()
            Q /= omega
            dQ /= omega
            ddQ /= omega

        E0 = eig[0]
        v = (eig[1]-eig[0]) * constants.E_h / constants.h / (constants.c * 100)
        print(v)
        #print(Q)

        F = -math.log(Q)/beta
        S = (E - F)/T
        Cv = (2/Q*dQ - T*pow(dQ/Q,2) + T/Q*ddQ)/beta

        return E, S, F, Q, Cv

    def get_symmetry_number(self):
        Log = QChemLog(os.path.join(self.directory,
            'ads{}.q.out'.format(self.nads)))
        symmetry = Log.load_conformer[0].modes[1].symmetry
        return symmetry

    def get_moment_of_inertia(self):
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
        #ads_conformer, unscaled_frequencies = adsLog.load_conformer()
        #inertia = ads_conformer.get_moment_of_inertia_tensor()
        #inertia_xyz = np.linalg.eigh(self.inertia)[1]
        print("HELLO")
        return Log.load_conformer()


def make_job(xyz, path, file_name, ncpus, charge=None, multiplicity=None, level_of_theory=None, basis=None, unrestricted=None, \
        is_QM_MM_INTERFACE=None, QM_USER_CONNECT=None, QM_ATOMS=None, force_field_params=None, fixed_molecule_string=None, opt=None, number_of_fixed_atoms=None):
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
