# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:57:57 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import snp_fct
import sys
sys.path.append('../')
import vector_fitting
import matrix_fitting
import fit_z
import fit_s


if __name__ == '__main__':
    s2p_file = 'two_coupled_SIW_ant_39GHz.s2p'
    freq, n, z, s, z0 = snp_fct.read_snp(s2p_file)
    s_50 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq*2j*np.pi
    for i in range(len(freq)):
        s_50[:, :, i] = np.matrix(z[:, :, i]/50-np.identity(n)) * np.linalg.inv(np.matrix(z[:, :, i]/50+np.identity(n)))
    
#    # Try matrix_fitting first
#    poles, residues, d, h = matrix_fitting.matrix_fitting(s_50, cs, n_poles=38, n_iters=20, has_h=0)
#    s_matrix_fit = matrix_fitting.matrix_model(cs, poles, residues, d, h)
#    
#    plt.figure()
#    plt.plot(freq, 20*np.log10(np.abs(s_50[0, 0, :])), 'b-')
#    plt.plot(freq, 20*np.log10(np.abs(s_matrix_fit[0, 0, :])), 'r--')
#    plt.plot(freq, 20*np.log10(np.abs(s_50[0, 0, :]-s_matrix_fit[0, 0, :])), 'k--')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Amplitude (dB)')
#    plt.figure()
#    plt.plot(freq, 20*np.log10(np.abs(s_50[1, 0, :])), 'b-')
#    plt.plot(freq, 20*np.log10(np.abs(s_matrix_fit[1, 0, :])), 'r--')
#    plt.plot(freq, 20*np.log10(np.abs(s_50[1, 0, :]-s_matrix_fit[1, 0, :])), 'k--')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Amplitude (dB)')
    
    # Fit even and odd mode separately
    s_even = ((s_50[0, 0, :] + s_50[0, 1, :] + s_50[1, 0, :] + s_50[1, 1, :])/2).reshape(len(freq))
    s_odd = ((s_50[0, 0, :] - s_50[0, 1, :] - s_50[1, 0, :] + s_50[1, 1, :])/2).reshape(len(freq))
    cs_all = np.logspace(10, 12, 1e5)*1j
    
    # Even mode
    poles_even, residues_even, d_even, h_even = fit_s.fit_s(s_even, cs, n_poles=22, n_iters=20, s_inf=-1)
    s_even_fit = vector_fitting.model(cs, poles_even, residues_even, d_even, h_even)
    
    plt.figure()
    plt.plot(freq, 20*np.log10(np.abs(s_even)), 'b-')
    plt.plot(freq, 20*np.log10(np.abs(s_even_fit)), 'r--')
    plt.plot(freq, 20*np.log10(np.abs(s_even-s_even_fit)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    # Odd mode
    poles_odd, residues_odd, d_odd, h_odd = fit_s.fit_s(s_odd, cs, n_poles=21, n_iters=20, s_inf=1)
    s_odd_fit = vector_fitting.model(cs, poles_odd, residues_odd, d_odd, h_odd)
    
    plt.figure()
    plt.plot(freq, 20*np.log10(np.abs(s_odd)), 'b-')
    plt.plot(freq, 20*np.log10(np.abs(s_odd_fit)), 'r--')
    plt.plot(freq, 20*np.log10(np.abs(s_odd-s_odd_fit)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    # Overall passivity
    s_even_fit_all = vector_fitting.model(cs_all, poles_even, residues_even, d_even, h_even)
    s_odd_fit_all = vector_fitting.model(cs_all, poles_odd, residues_odd, d_odd, h_odd)
    plt.figure()
    plt.plot(abs(cs_all)/2/np.pi, 20*np.log10(np.abs(s_even_fit_all)), 'r--')
    plt.plot(abs(cs_all)/2/np.pi, 20*np.log10(np.abs(s_odd_fit_all)), 'b--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    
    
#    # Fit z then convert to s
#    z_even = ((z[0, 0, :] + z[0, 1, :] + z[1, 0, :] + z[1, 1, :])/2).reshape(len(freq))
#    z_odd = ((z[0, 0, :] - z[0, 1, :] - z[1, 0, :] + z[1, 1, :])/2).reshape(len(freq))
#    cs_all = np.logspace(10, 12, 1e6)*1j
#    
#    # Even mode
#    poles_even, residues_even, d_even, h_even = fit_z.fit_z(z_even, cs, n_poles=24, n_iters=20, has_d=0, has_h=0)
#    z_even_fit = vector_fitting.model(cs, poles_even, residues_even, d_even, h_even)
#    
#    plt.figure()
#    plt.plot(freq, 20*np.log10(np.abs(z_even)), 'b-')
#    plt.plot(freq, 20*np.log10(np.abs(z_even_fit)), 'r--')
#    plt.plot(freq, 20*np.log10(np.abs(z_even-z_even_fit)), 'k--')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Amplitude (dB)')
#    
#    # Odd mode
#    poles_odd, residues_odd, d_odd, h_odd = fit_z.fit_z(z_odd, cs, n_poles=22, n_iters=20, has_d=0, has_h=0)
#    z_odd_fit = vector_fitting.model(cs, poles_odd, residues_odd, d_odd, h_odd)
#    
#    plt.figure()
#    plt.plot(freq, 20*np.log10(np.abs(z_odd)), 'b-')
#    plt.plot(freq, 20*np.log10(np.abs(z_odd_fit)), 'r--')
#    plt.plot(freq, 20*np.log10(np.abs(z_odd-z_odd_fit)), 'k--')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Amplitude (dB)')
#    
#    # Overall passivity
#    z_even_fit_all = vector_fitting.model(cs_all, poles_even, residues_even, d_even, h_even)
#    z_odd_fit_all = vector_fitting.model(cs_all, poles_odd, residues_odd, d_odd, h_odd)
#    plt.figure()
#    plt.plot(abs(cs_all)/2/np.pi, 20*np.log10(np.abs(z_even_fit_all.real)), 'r--')
#    plt.plot(abs(cs_all)/2/np.pi, 20*np.log10(np.abs(z_odd_fit_all.real)), 'b--')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Amplitude (dB)')
    
    