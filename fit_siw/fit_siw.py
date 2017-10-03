# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:57:57 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import sys
sys.path.append('../')
import vector_fitting

## import from a specific directory
#import importlib.util
#spec = importlib.util.spec_from_file_location("vectorFitting", "C:\\Users\\kirknie\\Documents\\Python\\Vector_fitting\\vectorFitting.py")
#vectorFitting = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(vectorFitting)


def get_z0(s1p_file):
    z0 = []
    with open(s1p_file) as f:
        for l in f:
            if '! Port Impedance' in l:
                z0_str = list(filter(None, l[len('! Port Impedance'):].split(' ')))
                z0.append(float(z0_str[0]) + 1j*float(z0_str[1]))
    return z0

if __name__ == '__main__':
    s1p_file = 'SIW_ant_39GHz_from_20GHz_to_50GHz.s1p'
    ant_data = rf.Network(s1p_file)
    freq = ant_data.f[500:]
    cs = freq*2j*np.pi
    s_data = ant_data.s.reshape(len(ant_data))[500:]
    z0_data = np.array(get_z0(s1p_file)[500:])
    z_data = z0_data * (1+s_data) / (1-s_data)
    s_data = (z_data-50) / (z_data+50)
    z0 = 50
    
    poles, residues, d, h = vector_fitting.vector_fitting_rescale(z_data, cs, n_poles=18, n_iters=20, has_d=1, has_h=1)
    f_fit = vector_fitting.model(cs, poles, residues, d, h)
    
    print(poles)
    
    plt.figure()
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(z_data)), 'b-')
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(f_fit)), 'r--')
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(f_fit-z_data)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    # plot the fitted data in a broader bandwidth
    s_all = np.logspace(10, 12, 1e5)*1j
    f_fit_all = vector_fitting.model(s_all, poles, residues, d, h)
    
    #z = vector_fitting.calculate_zeros(poles, residues, d)
    #print('Zeros are:', z)
    #check_zeros = vector_fitting.model(z, poles, residues, d, h)
    
    # For S, zeros are z = z0, poles are z = -z0
    s_zeros = vector_fitting.calculate_zeros(poles, residues, d-z0)
    s_poles = vector_fitting.calculate_zeros(poles, residues, d+z0)
    
    bound = -np.pi/2*(sum(s_poles)+sum(s_zeros))
    print('Bound is {:.5e}'.format(bound.real))
    
    plt.figure()
    plt.semilogx(np.abs(s_all), 20*np.log10(np.abs(f_fit_all)), 'b-')
    plt.semilogx(np.abs(s_all), 20*np.log10(np.abs(f_fit_all.real)), 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.show()
    
    