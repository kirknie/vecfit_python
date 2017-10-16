# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:19:55 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vector_fitting


def fit_z_step(f, s, poles, has_d=1, has_h=1, fixed_poles=[], reflect_z=[], pole_wt=0):
    # Function generates a new set of poles
    # Should create a class and let z_fitting inherit the class
    
    # First group input poles into complex conjugate pairs
    poles_all = np.concatenate([poles, fixed_poles])
    nP = len(poles_all)
    nP_iter = len(poles)
    nF = len(s)
    poles_pair = vector_fitting.pair_poles(poles_all)
    
    # Constraint: d = sum(-1/2*( 1/(s0-ai)+1/(-s0-ai) ))
    if reflect_z:
        has_d = 0
    
    # Then generate the A and b from Appendix A
    A = np.zeros((nF, nP+has_d+has_h+nP_iter), dtype=np.complex64)
    for i, p in enumerate(poles_all):
        if poles_pair[i] == 0:
            A[:, i] = 1/(s-p)
        elif poles_pair[i] == 1:
            A[:, i] = 1/(s-p) + 1/(s-p.conjugate())
        elif poles_pair[i] == 2:
            A[:, i] = 1j/(s-p) - 1j/(s-p.conjugate())
        else:
            raise RuntimeError("poles_pair[%d] = %d" % (i, poles_pair[i]))
        if i < nP_iter:
            A[:, -nP_iter+i] = -A[:, i] * f
    if has_d:
        A[:, nP] = 1
    if has_h:
        A[:, nP+has_d] = s
    if reflect_z:
        s0 = reflect_z[0]
        for i, p in enumerate(poles_all):
            if poles_pair[i] == 0:
                A[:, i] += -( 1/(s0-p) + 1/(-s0-p) ) / 2
            elif poles_pair[i] == 1:
                A[:, i] += -( 1/(s0-p) + 1/(-s0-p) ) / 2 - ( 1/(s0-p.conjugate()) + 1/(-s0-p.conjugate()) ) / 2
            elif poles_pair[i] == 2:
                A[:, i] += -1j*( 1/(s0-p) + 1/(-s0-p) ) / 2 + 1j*( 1/(s0-p.conjugate()) + 1/(-s0-p.conjugate()) ) / 2
    b = f
    
    # Solve for x in (A.8) using least mean square
    A = np.vstack([np.real(A), np.imag(A)])
    b = np.concatenate([np.real(b), np.imag(b)])
    
    if pole_wt > 0:  # Put some weight on the sum of poles
        A = np.vstack([A, np.zeros((1, nP+has_d+has_h+nP_iter), dtype=np.complex64)])
        b = np.concatenate([b, [np.real(np.sum(poles))*pole_wt]])
        for i, p in enumerate(poles):
            if poles_pair[i] == 0:
                A[-1, -nP_iter+i] = 1*pole_wt
            elif poles_pair[i] == 1:
                A[-1, -nP_iter+i] = 2*pole_wt
            elif poles_pair[i] == 2:
                A[-1, -nP_iter+i] = 0*pole_wt
    
    
    
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)
    
    #residues = x[:nP]
    #d = x[nP]
    #h = x[nP+1]
    
    # Calculate the new poles by following Appendix B
    A = np.diag(poles)
    b = np.ones(nP_iter)
    for i, p in enumerate(poles):
        if poles_pair[i] == 1:
            A[i, i] = A[i+1, i+1] = p.real
            A[i, i+1] = -p.imag
            A[i+1, i] = p.imag
            b[i] = 2
            b[i+1] = 0
    c = x[-nP_iter:]
    
    H = A - np.outer(b, c)
    #print('H is:', H)
    H = np.real(H) # H should already be real
    new_poles = np.sort(np.linalg.eigvals(H)) # sorted new_poles should have [real, complex pairs]
    #print('Old poles are:', poles, 'New poles are:', new_poles)
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    
    # Return new poles
    return new_poles

def calculate_residues_z(f, s, poles, has_d=1, has_h=1, reflect_z=[]):
    # Function uses the input poles to calculate residues
    
    # First group input poles into complex conjugate pairs
    nP = len(poles)
    nF = len(s)
    poles_pair = vector_fitting.pair_poles(poles)
    
    if reflect_z:
        has_d = 0
    
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
    if reflect_z:
        s0 = reflect_z[0]
        for i, p in enumerate(poles):
            if poles_pair[i] == 0:
                A[:, i] += -( 1/(s0-p) + 1/(-s0-p) ) / 2
            elif poles_pair[i] == 1:
                A[:, i] += -( 1/(s0-p) + 1/(-s0-p) ) / 2 - ( 1/(s0-p.conjugate()) + 1/(-s0-p.conjugate()) ) / 2
            elif poles_pair[i] == 2:
                A[:, i] += -1j*( 1/(s0-p) + 1/(-s0-p) ) / 2 + 1j*( 1/(s0-p.conjugate()) + 1/(-s0-p.conjugate()) ) / 2
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
    if reflect_z:
        s0 = reflect_z[0]
        d = np.sum([-(r/(s0-p)+r/(-s0-p))/2 for p, r in zip(poles, residues)])
    
    return residues, d, h


def fit_z(f, s, n_poles=10, n_iters=10, has_d=1, has_h=1, fixed_poles=[], reflect_z=[], pole_wt=0):
    # Function runs vector fitting
    # Assume w is imaginary, non-negative and in a ascending order
    # reflect_z needs to be in conjugate pairs and in the RHP, only support 1 reflection point now
    if reflect_z and not has_d:
        raise RuntimeError('Cannot guarantee reflection when d=0')
    
    w = np.imag(s)
    poles = vector_fitting.init_poles(w[-1], n_poles)
    #print(poles)
    #for p in poles:
        #print('Poles:', p)
    
    for loop in range(n_iters):
        poles = fit_z_step(f, s, poles, has_d=has_d, has_h=has_h, fixed_poles=fixed_poles, reflect_z=reflect_z, pole_wt=pole_wt)
        #print(poles)
    
    poles = np.concatenate([poles, fixed_poles])
    residues, d, h = calculate_residues_z(f, s, poles, has_d=has_d, has_h=has_h, reflect_z=reflect_z)
    
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