#/usr/bin env

# import needed modules
import os
from ape.sampling import SamplingJob
from ape.qchem import QChemLog
from ape.common import get_electronic_energy
from ape.InternalCoordinates import getXYZ
from tnuts.qchem import get_level_of_theory
from rigidObject import RigidObject
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import scipy.special
from sympy.physics.wigner import gaunt

class Adsorbate:
    def __init__(self, freqfile, directory, nads, T=300):
        self.freqfile = freqfile
        self.directory = directory
        self.nads = nads
        self.T = T
        
        self.label = freqfile.split('.')[0]
        self.Log = QChemLog(os.path.join(directory, freqfile))
        level_of_theory_kwargs = get_level_of_theory(Log)
        
        self.samp = SamplingJob(
            input_file=os.path.join(directory, freqfile),
            label=label,
            ncpus=4, output_directory=directory,
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
        
        # QM kwargs
        self.qchem_kwargs['is_QM_MM_INTERFACE'] = self.samp.is_QM_MM_INTERFACE
        self.qchem_kwargs['QM_USER_CONNECT'] = self.samp.QM_USER_CONNECT
        self.qchem_kwargs['QM_ATOMS'] = self.samp.QM_ATOMS
        self.qchem_kwargs['force_field_params'] = self.samp.force_field_params
        self.qchem_kwargs['fixed_molecule_string'] = self.samp.fixed_molecule_string
        self.qchem_kwargs['opt'] = self.samp.opt
        self.qchem_kwargs['number_of_fixed_atoms'] = self.samp.number_of_fixed_atoms
        
        # get mass of adsorbate
        mass = self.samp.conformer.mass.value_si[:self.nads]
        coordinates = self.samp.conformer.coordinates.value[:self.nads]
        #if linear is None:
        #    linear = is_linear(coordinates)
        #    if linear:
        #        logging.info('Determined species {0} to be linear.'.format(label))
        
        # get moments of inertia of adsorbate
        #self.adsorbate = load_adsorbate(Log) # OF THE ADSORBATE
        
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
        
        self.com = np.array([xm,ym,zm])    # AGAIN, THIS WILL STAY CONSTANT
        self.rigidrotor = RigidObject(nads, self.samp.internal.c3d)
        self.rigidrotor.change_origin(self.com) # Set rotational origin to c.o.m.
        
        self.record_script ='''{natom}\n# Point {sample} Energy = {e_elect}\n{xyz}\n'''
    
    def reorient_by(self, dtheta, dphi):
        # theta is polar angle, phi is azimuthal angle
        self.rigidrotor.rotate_x(dphi)
        self.rigidrotor.rotate_z(dtheta)
        self.samp.internal.cart_coords[:(3*self.nads)] = self.rigidrotor.to_array().flatten()
    
    def sample(self, na=20, dry=False, write=False):
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
        
        da=np.pi/(na-1); # latitude angle step
        a = np.pi/2
        actual_a = 0
        for ia in range(na): # slice sphere to circles in xy planes
            r=np.cos(a);                           # radius of actual circle in xy plane
            z=np.sin(a);                           # height of actual circle in xy plane
            nb=np.ceil(2.0*np.pi*r/da)
            db=2.0*np.pi/nb;             # longitude angle step
            if ia==0 or ia==na-1:
                nb=1
                db=0.0 # handle edge cases
            b = 0.0
            for ib in range(int(nb)):  # cut circle to vertexes
                x=r*np.cos(b);                     # compute x,y of vertex
                y=r*np.sin(b);
                grid.append(np.array([x,y,z]))
                sph_grid.append(np.array([actual_a,b]))
                self.reorient_by(da,db)
                xyz = getXYZ(self.samp.symbols, self.samp.internal.cart_coords)
                
                # run Q-Chem job
                if not dry:
                    E = get_electronic_energy(xyz=xyz, path=self.path, file_name=self.file_name.format(count),
                                             **self.qchem_kwargs)
                    v.append(E)
                else:
                    E = 0
                    
                if write:
                    # write a file showing geometric configurations to sample
                    name = 'configs.txt'
                    with open(os.path.join(self.directory, name), 'a') as f:
                        content = self.record_script.format(natom=self.samp.natom,
                                                           sample=count, e_elect=E, xyz=xyz)
                        f.write(content)
                        
                count += 1
                b += db
            a -= da
            actual_a += da
        if not dry:
            v = np.array(v)-np.min(v)
            #np.save
        return np.array(grid), np.array(sph_grid), np.array(v)
