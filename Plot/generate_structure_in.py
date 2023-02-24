# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:53:14 2023

@author: MOJUE ZHANG
"""

import numpy as np
from pyevtk.hl import gridToVTK



def read_file(Nx, Ny, Nz,step):
    filename = 'eta_index%d.txt' % step
    indata = np.loadtxt(filename, skiprows = 1)
    eta_index = np.zeros((Nx, Ny, Nz))
    for i in range(len(indata)):
        eta_index[int(indata[i,0]), int(indata[i, 1]), int(indata[i,2])] = indata[i, 3]
    
    return eta_index
    
    
def generate_structure_file(Nx, Ny, Nz,step):
    filename = 'eta_index%d.txt' % step
    indata = np.loadtxt(filename)
    # modify the last column of the array
    last_column = indata[:, -1]
    last_column[last_column != 0] = 1
    indata[:, -1] = last_column
    outputfile = 'struct_%d.in' % step
    np.savetxt(outputfile, indata, fmt="%d")
    return indata



def generate_orient_random_file(Norient):
    filename = 'orientation.txt'
    # Generate column 1 values
    col1 = np.random.uniform(low=0, high=2*np.pi, size=(Norient, 1))    
    # Generate column 2 values
    col2 = np.random.uniform(low=0, high=np.pi, size=(Norient, 1))    
    # Generate column 3 values
    col3 = np.random.uniform(low=0, high=2*np.pi, size=(Norient, 1))    
    # Concatenate columns into a single array
    data = np.concatenate((col1, col2, col3), axis=1)    
    # Save array as a text file
    np.savetxt(filename, data)
    return data



def generate_eulerAng_file(eta_index, orientation, step):
    filename = 'eulerAng_%d.in' % step
    eulerAng = np.zeros((64, 64, 64, 3))
    
    
    # get the orientation numbers from the last column of the struct_data array
    orientation_nums = eta_index.astype(int)
    zero_indices = np.where(orientation_nums == 0)
    eulerAng[zero_indices[0], zero_indices[1], zero_indices[2], :] = [0, 0, 0]
    
    # set the Euler angles to the corresponding values for non-zero orientation numbers
    nonzero_indices = np.where(orientation_nums > 0)
    orient_indices = orientation_nums[nonzero_indices] - 1
    eulerAng[nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], :] = orientation[orient_indices, :]
    
    # save the eulerang.in file
    eulerAng_data_flat = eulerAng.reshape((64*64*64, 3))
    index_data = np.indices(eulerAng.shape[:-1]).reshape((3, -1)).T
    eulerAng_data_with_index = np.hstack((index_data, eulerAng_data_flat))
    np.savetxt(filename, eulerAng_data_with_index, fmt='%d %d %d %f %f %f')
    
    return eulerAng_data_with_index
    
 

Nx = 64; Ny = 64; Nz = 64
Norient = 36;
Nstep = 200; Noutput = 40

#orientation = generate_orient_random_file(Norient)
orientation = np.loadtxt('orientation.txt', skiprows = 1)

for i in range(40, Nstep + Noutput, Noutput):
    eta_index =read_file(Nx, Ny, Nz, i)
    structure = generate_structure_file(Nx, Ny, Nz,i)
    eulerAng_data_with_index = generate_eulerAng_file(eta_index, orientation, i)
