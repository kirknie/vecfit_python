# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:34:57 2018

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
import bound_fct

## import from a specific directory
#import importlib.util
#spec = importlib.util.spec_from_file_location("vectorFitting", "C:\\Users\\kirknie\\Documents\\Python\\Vector_fitting\\vectorFitting.py")
#vectorFitting = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(vectorFitting)


if __name__ == '__main__':
    s1p_file = 'SIW_system_50mil.s1p'
    freq, n, z_data, s_siw, z0_data = snp_fct.read_snp(s1p_file)
    
#    freq = freq[500:1501]
#    s_siw = s_siw[500:1501]
#    z0_data = z0_data[500:1501]
#    z_data = z_data[500:1501]
    #z0 = 50
    z0 = 120
    s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi
    s_all = np.logspace(10, 12, 1e5)*1j
    
    # Try to fit S
    #poles, residues, d, h = fit_s.fit_s(s_data, cs, n_poles=41, n_iters=20, s_dc=0, s_inf=1, pole_wt=0, bound_wt=1.2e-13)
    poles, residues, d, h = fit_s.fit_s(s_data, cs, n_poles=41, n_iters=20, s_dc=0, s_inf=1, pole_wt=0)
    fs_fit = vector_fitting.model(cs, poles, residues, d, h)
    s_zeros = vector_fitting.calculate_zeros(poles, residues, d)
    fs_fit_all = vector_fitting.model(s_all, poles, residues, d, h)
    
    bound, bw = bound_fct.bound_s(poles, residues, d, 0, f0=39e9)
    print('Bound is {:.5e}'.format(bound))
    print('BW is {:.5e}'.format(bw))
    
    bound, bw = bound_fct.bound_s(poles, residues, d, np.inf)
    print('Bound is {:.5e}'.format(bound))
    print('BW is {:.5e}'.format(bw))
    
    bound_error = bound_fct.bound_error_s(s_data, cs, poles, residues, d, np.inf)
    print('Bound error is {:.5e}'.format(bound_error))
    
    ant_integral = bound_fct.bound_integral(cs.imag, np.abs(s_data), np.inf)
    print('The inegral of the antenna is {:.5e}'.format(ant_integral))
    
    #bound_fct.plot_improved_bound(poles, residues, d, 1.2e10, 4e11)
    
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
    
    print('check s', max(np.abs(fs_fit_all)))
    