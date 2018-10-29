# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:19:55 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vector_fitting


def fit_z(f, s, n_poles=10, n_iters=10, has_d=1, has_h=1, fixed_poles=[], reflect_z=[]):
    poles, residues, d, h = vector_fitting.vector_fitting_rescale(f, s, n_poles=n_poles, n_iters=n_iters, has_d=has_d, has_h=has_h, fixed_poles=fixed_poles, reflect_z=reflect_z)
    return poles, residues, d, h


if __name__ == '__main__':
    """
    Specify reflection point when fitting z
    """
    
    s_test = 1j*np.linspace(1e3, 1e5, 800)
    poles_test = [-4500,
                  -41000,
                  -100+5000j, -100-5000j,
                  -120+15000j, -120-15000j,
                  -3000+35000j, -3000-35000j,
                  -200+45000j, -200-45000j,
                  -1500+45000j, -1500-45000j,
                  -500+70000j, -500-70000j,
                  -1000+73000j, -1000-73000j,
                  -2000+90000j, -2000-90000j]
    residues_test = [-3000,
                     -83000,
                     -5+7000j, -5-7000j,
                     -20+18000j, -20-18000j,
                     6000+45000j, 6000-45000j,
                     40+60000j, 40-60000j,
                     90+10000j, 90-10000j,
                     50000+80000j, 50000-80000j,
                     1000+45000j, 1000-45000j,
                     -5000+92000j, -5000-92000j]
    d_test = 19.07733751
    h_test = 5e-4
    
    f_test = vector_fitting.model(s_test, poles_test, residues_test, d_test, h_test)
    
    poles, residues, d, h = fit_z(f_test, s_test, n_poles=17, n_iters=10, has_d=1, has_h=1, reflect_z=[0])
    f_fit = vector_fitting.model(s_test, poles, residues, d, h)
    
    plt.figure()
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_test)), 'b-')
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_fit)), 'r--')
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_fit-f_test)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    s_all = 1j*np.linspace(1, 1e6, 2000)
    f_all = vector_fitting.model(s_all, poles, residues, d, h)
    plt.figure()
    plt.plot(np.abs(s_all), 20*np.log10(np.abs(f_all)), 'b-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    plt.show()