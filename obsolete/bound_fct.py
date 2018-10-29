# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:31:51 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vector_fitting


def f_integral(w, reflect):
    if reflect == np.inf:
        f = np.ones(len(w))
    elif reflect.real > 0.0:
        f = 1/2 * (1/(reflect - 1j*w) + 1/(reflect + 1j*w)).real
    elif reflect.real == 0.0:
        f = 1/2 * (np.power(reflect.imag - w, -2) + np.power(reflect.imag - w, -2))
    else:
        raise RuntimeError('Invalid reflection point!')
    return f

def bound_s(poles, residues, d, reflect, tau=0.2, f0=0):
    bw = np.nan
    zeros = vector_fitting.calculate_zeros(poles, residues, d)
    if reflect == 0:
        bound = -np.pi/2*(sum(1/poles)+sum(1/zeros))
        bound = bound.real
        if f0 != 0:
            bw = bound/2/np.pi/np.log(1/tau)*(2*np.pi*f0)**2
    elif reflect == np.inf:
        bound = -np.pi/2*(sum(poles)+sum(zeros))
        bound = bound.real
        bw = bound/2/np.pi/np.log(1/tau)
    else:
        raise RuntimeError('Bound calculation unsopported yet!')
    return bound, bw

def bound_error_s(f, s, poles, residues, d, reflect, tau=0.2):
    f_fit = vector_fitting.model(s, poles, residues, d, 0)
    f_error = f_fit - f
    # Calculate rho first (single load)
    rho = (2*np.abs(f_error))/(1-np.power(np.abs(f), 2))
    int_fct = f_integral(s.imag, reflect) / 2 * np.log(1 + (1-tau**2)/tau**2 * rho)
    # delta_b = np.sum((int_fct[:-1] + int_fct[1:]) / 2 * (s.imag[1:] - s.imag[:-1]))
    delta_b = num_integral(s.imag, int_fct)
    return delta_b

def num_integral(x, y):
    return np.sum((y[:-1] + y[1:]) / 2 * (x[1:] - x[:-1]))

def bound_integral(w, r, reflect):
    f = f_integral(w, reflect) * np.log(1/r)
    return num_integral(w, f)
    
def plot_improved_bound(poles, residues, d, real_limit, imag_limit):
    sample = 2000
    x = np.linspace(-real_limit, 0, sample)
    y = np.linspace(-imag_limit, imag_limit, 2*sample)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    s = xv + 1j*yv
    f = vector_fitting.model(s, poles, residues, d, 0)
    
    # do the plot
    plt.figure()
    plt.contourf(x, y, 2-np.abs(f), [1, 2])
    
    zeros = vector_fitting.calculate_zeros(poles, residues, d)
    plt.plot(poles.real, poles.imag, 'x')
    plt.plot(zeros.real, zeros.imag, 'o')
    plt.axis([-real_limit, 0, -imag_limit, imag_limit])
    plt.xlabel('Re\{s\}')
    plt.ylabel('Im\{s\}')
    
    return s, f
