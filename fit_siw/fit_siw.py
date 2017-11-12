# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:57:57 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import snp_fct
import sys
sys.path.append('../')
import vector_fitting
import fit_z
import fit_s

## import from a specific directory
#import importlib.util
#spec = importlib.util.spec_from_file_location("vectorFitting", "C:\\Users\\kirknie\\Documents\\Python\\Vector_fitting\\vectorFitting.py")
#vectorFitting = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(vectorFitting)


if __name__ == '__main__':
    #s1p_file = 'SIW_ant_39GHz_from_20GHz_to_50GHz.s1p'
    s1p_file = 'single_SIW_antenna_39GHz_50mil.s1p'
    freq, n, z_data, s_data, z0_data = snp_fct.read_snp(s1p_file)
    
    freq = freq[500:]
    s_data = s_data[500:]
    z0_data = z0_data[500:]
    z_data = z_data[500:]
    s_data = (z_data-50) / (z_data+50)
    cs = freq*2j*np.pi
    z0 = 50
    
    #poles, residues, d, h = vector_fitting.vector_fitting_rescale(z_data, cs, n_poles=20, n_iters=20, has_d=1, has_h=0, fixed_poles=[0])
    poles, residues, d, h = fit_z.fit_z(z_data, cs, n_poles=14, n_iters=20, has_d=1, has_h=0)
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
    
    bound = -np.pi/2*(sum(1/s_poles)+sum(1/s_zeros))
    print('Bound is {:.5e}'.format(bound.real))
    print('BW is {:.5e}'.format(bound.real/2/np.pi/np.log(1/0.2)*(2*np.pi*39e9)**2))
    
    plt.figure()
    plt.semilogx(np.abs(s_all), 20*np.log10(np.abs(f_fit_all)), 'b-')
    plt.semilogx(np.abs(s_all), 20*np.log10(np.abs(f_fit_all.real)), 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    s_fit_all = (f_fit_all-50) / (f_fit_all+50)
    plt.figure()
    plt.semilogx(np.abs(s_all), 20*np.log10(np.abs(s_fit_all)), 'b-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    
    # Try to fit S
    poles, residues, d, h = fit_s.fit_s(s_data, cs, n_poles=19, n_iters=20, s_dc=0, s_inf=1, pole_wt=1e-12)
    fs_fit = vector_fitting.model(cs, poles, residues, d, h)
    s_zeros = vector_fitting.calculate_zeros(poles, residues, d)
    fs_fit_all = vector_fitting.model(s_all, poles, residues, d, h)
    bound = -np.pi/2*(sum(poles)+sum(s_zeros))
    print('Bound is {:.5e}'.format(bound.real))
    print('BW is {:.5e}'.format(bound.real/2/np.pi/np.log(1/0.2)))
    
    plt.figure()
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(s_data)), 'b-')
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(fs_fit)), 'r--')
    plt.plot(np.abs(cs)/2/np.pi, 20*np.log10(np.abs(fs_fit-s_data)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.figure()
    plt.plot(np.abs(s_all)/2/np.pi, 20*np.log10(np.abs(fs_fit_all)), 'b-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.show()
    
    