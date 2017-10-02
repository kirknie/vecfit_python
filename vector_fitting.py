# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""


"""
Implementation of vector fitting under Python.  RF loads such as antennas can be fitted with this toolbox so that the Bode-Fano bounds can be applied.
"""

import numpy as np
import matplotlib.pyplot as plt
#import sympy


def model(s, poles, residues, d, h):
    return sum(r/(s-p) for (p, r) in zip(poles, residues)) + d + s*h

def init_poles(w, n_poles):
    loss_ratio = 1e-2
    n_poles = int(n_poles)
    
    if n_poles == 1:
        poles = np.array(-loss_ratio*w[-1])
    elif n_poles == 2:
        poles = w[-1]*np.array([-loss_ratio-1j, -loss_ratio+1j])
    elif n_poles == 3:
        poles = w[-1]*np.array([loss_ratio, -loss_ratio-1j, -loss_ratio+1j])
    elif n_poles > 3 and n_poles%2 == 0:
        complex_poles = np.linspace(0, 1, int((n_poles-2)/2+1))[1:]
        poles = np.concatenate( [[p*(-loss_ratio-1j), p*(-loss_ratio+1j)] for p in complex_poles] )
        poles = w[-1]*np.concatenate( [[-1, -10], poles] )
    elif n_poles > 3 and n_poles%2 == 1:
        complex_poles = np.linspace(-1, 1, int((n_poles-3)/2+1))[1:]
        poles = np.concatenate( [[p*(-loss_ratio-1j), p*(-loss_ratio+1j)] for p in complex_poles] )
        poles = w[-1]*np.concatenate( [[-1, -3, -10], poles] )
    else:
        raise RuntimeError('Error: invalid number of poles')
    print('Number of poles', len(poles))
    return poles

def pair_poles(poles):
    # Function groups poles into complex conjugate pairs
    poles_pair = np.zeros(len(poles))
    
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or poles_pair[i-1] != 1:
                if p.conjugate() != poles[i+1]:
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s, %s" % (p, poles[i+1]))
                poles_pair[i] = 1
            else:
                poles_pair[i] = 2
    
    return poles_pair

def vector_fitting_step(f, s, poles, has_d=1, has_h=1):
    # Function generates a new set of poles
    
    # First group input poles into complex conjugate pairs
    nP = len(poles)
    nF = len(s)
    poles_pair = pair_poles(poles)
    
    # Then generate the A and b from Appendix A
    A = np.zeros((nF, 2*nP+has_d+has_h), dtype=np.complex64)
    for i, p in enumerate(poles):
        if poles_pair[i] == 0:
            A[:, i] = 1/(s-p)
        elif poles_pair[i] == 1:
            A[:, i] = 1/(s-p) + 1/(s-p.conjugate())
        elif poles_pair[i] == 2:
            A[:, i] = 1j/(s-p) - 1j/(s-p.conjugate())
        else:
            raise RuntimeError("poles_pair[%d] = %d" % (i, poles_pair[i]))
        A[:, -nP+i] = -A[:, i] * f
    if has_d:
        A[:, nP] = 1
    if has_h:
        A[:, nP+has_d] = s
    
    b = f
    
    # Solve for x in (A.8) using least mean square
    A = np.vstack([np.real(A), np.imag(A)])
    b = np.concatenate([np.real(b), np.imag(b)])
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)
    
    #residues = x[:nP]
    #d = x[nP]
    #h = x[nP+1]
    
    # Calculate the new poles by following Appendix B
    A = np.diag(poles)
    b = np.ones(nP)
    for i, p in enumerate(poles):
        if poles_pair[i] == 1:
            A[i, i] = A[i+1, i+1] = p.real
            A[i, i+1] = -p.imag
            A[i+1, i] = p.imag
            b[i] = 2
            b[i+1] = 0
    c = x[-nP:]
    
    H = A - np.outer(b, c)
    #print('H is:', H)
    H = np.real(H) # H should already be real
    new_poles = np.sort(np.linalg.eigvals(H)) # sorted new_poles should have [real, complex pairs]
    #print('Old poles are:', poles, 'New poles are:', new_poles)
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    
    # Return new poles
    return new_poles

def calculate_residues(f, s, poles, has_d=1, has_h=1):
    # Function uses the input poles to calculate residues
    
    # First group input poles into complex conjugate pairs
    nP = len(poles)
    nF = len(s)
    poles_pair = pair_poles(poles)
    
    # Then generate the A and b from Appendix A, without the negative part
    A = np.zeros((nF, nP+has_d+has_h), dtype=np.complex64)
    for i, p in enumerate(poles):
        if poles_pair[i] == 0:
            A[:, i] = 1/(s-p)
        elif poles_pair[i] == 1:
            A[:, i] = 1/(s-p) + 1/(s-p.conjugate())
        elif poles_pair[i] == 2:
            A[:, i] = 1j/(s-p) - 1j/(s-p.conjugate())
        else:
            raise RuntimeError("poles_pair[%d] = %d" % (i, poles_pair[i]))
    if has_d:
        A[:, nP] = 1
    if has_h:
        A[:, nP+has_d] = s
    
    b = f
    
    # Solve for x in (A.8) using least mean square
    A = np.vstack([np.real(A), np.imag(A)])
    b = np.concatenate([np.real(b), np.imag(b)])
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)
    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning!: Ill Conditioned Matrix. Consider scaling the problem down')
    print('Cond(A):', cA)
    
    residues = np.complex64(x[:nP])
    for i, pp in enumerate(poles_pair):
       if pp == 1:
           r1, r2 = residues[i:i+2]
           residues[i] = r1 - 1j*r2
           residues[i+1] = r1 + 1j*r2
    d = 0
    h = 0
    if has_d:
        d = x[nP]
    if has_h:
        h = x[nP+has_d]
    
    return residues, d, h


def calculate_zeros(poles, residues, d):
    # First group input poles into complex conjugate pairs
    nP = len(poles)
    poles_pair = pair_poles(poles)
    
    # Then calculate the zeros by following Appendix B
    A = np.diag(poles)
    b = np.ones(nP)
    c = residues/d
    for i, p in enumerate(poles):
        if poles_pair[i] == 1:
            A[i, i] = A[i+1, i+1] = p.real
            A[i, i+1] = p.imag
            A[i+1, i] = -p.imag
            b[i] = 2
            b[i+1] = 0
            c[i+1] = c[i].imag
            c[i] = c[i].real
    
    H = A - np.outer(b, c)
    #print('H is:', H)
    H = H.real # H should already be real
    z = np.sort(np.linalg.eigvals(H)) # sorted zeros should have [real, complex pairs]
    return z


def vector_fitting(f, s, n_poles=10, n_iters=10, has_d=1, has_h=1):
    # Function runs vector fitting
    # Assume w is imaginary, non-negative and in a ascending order
    w = np.imag(s)
    poles = init_poles(w, n_poles)
    #print(poles)
    
    for loop in range(n_iters):
        poles = vector_fitting_step(f, s, poles, has_d=has_d, has_h=has_h)
        #print(poles)
    
    residues, d, h = calculate_residues(f, s, poles, has_d=has_d, has_h=has_h)
    
    return poles, residues, d, h

def vector_fitting_rescale(f, s, **kwargs):
    # Function rescales f and s and run vector fitting
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])
    
    poles_scaled, residues_scaled, d_scaled, h_scaled = vector_fitting(f/f_scale, s/s_scale, **kwargs)
    poles = poles_scaled * s_scale
    residues = residues_scaled * f_scale * s_scale
    d = d_scaled * f_scale
    h = h_scaled * f_scale / s_scale
    
    return poles, residues, d, h


if __name__ == '__main__':
    s_test = 1j*np.linspace(1, 1e5, 800)
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
    d_test = .2
    h_test = 5e-4
    
    f_test = model(s_test, poles_test, residues_test, d_test, h_test)
    
    #poles, residues, d, h = vector_fitting(fTest, sTest, nIters=2)
    poles, residues, d, h = vector_fitting_rescale(f_test, s_test, n_poles=16, n_iters=10, has_d=1, has_h=1)
    f_fit = model(s_test, poles, residues, d, h)
    
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_test)), 'b-')
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_fit)), 'r--')
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_fit-f_test)), 'k--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.show()


