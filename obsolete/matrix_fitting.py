# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:14:44 2017

@author: kirknie
"""



import numpy as np
import matplotlib.pyplot as plt
import vector_fitting
#import sympy


def matrix_model(s, poles, residues, d, h):
    nD = np.size(residues, 0)
    #nP = len(poles)
    nF = len(s)
    f = np.zeros([nD, nD, nF], dtype=np.complex128)
    for i, si in enumerate(s):
        f[:, :, i] = np.sum(residues[:, :, j]/(si-p) for j, p in enumerate (poles)) + d + si*h
    return f


def matrix_fitting_step(f, s, poles, has_d=1, has_h=1):
    # Function generates a new set of poles
    
    # First group input poles into complex conjugate pairs
    nP = len(poles)
    nF = len(s)
    nD = np.size(f, 0) # dimension of vector
    poles_pair = vector_fitting.pair_poles(poles)
    
    # Then generate the A and b from Appendix A
    A = np.zeros((nF*nD, (nP+has_d+has_h)*nD+nP), dtype=np.complex128)
    A1 = np.zeros((nF, nP+has_d+has_h), dtype=np.complex128) # for residues and d, h
    for i, p in enumerate(poles):
        if poles_pair[i] == 0:
            A1[:, i] = 1/(s-p)
        elif poles_pair[i] == 1:
            A1[:, i] = 1/(s-p) + 1/(s-p.conjugate())
        elif poles_pair[i] == 2:
            A1[:, i] = 1j/(s-p) - 1j/(s-p.conjugate())
        else:
            raise RuntimeError("poles_pair[%d] = %d" % (i, poles_pair[i]))
    if has_d:
        A1[:, nP] = 1
    if has_h:
        A1[:, nP+has_d] = s

    for i in range(nD):
        A[nF*i:nF*(i+1), (nP+has_d+has_h)*i:(nP+has_d+has_h)*(i+1)] = A1[:, :]
        for j in range(nP):
            A[nF*i:nF*(i+1), -nP+j] = -A1[:, j] * f[i, :]
    
    b = f.reshape(nF*nD) # order in rows (nF)
    
    # Solve for x in (A.8) using least mean square
    A = np.vstack([np.real(A), np.imag(A)])
    b = np.concatenate([np.real(b), np.imag(b)])
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    
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
    #print('New poles are:', new_poles)
    unstable = np.real(new_poles) > 0
    new_poles[unstable] -= 2*np.real(new_poles)[unstable]
    
    # Return new poles
    return new_poles


def calculate_matrix_residues(f, s, poles, has_d=1, has_h=1):
    # Function calls calculate_residues nD times
    nP = len(poles)
    nF = len(s)
    nD = np.size(f, 0) # dimension of vector
    residues = np.zeros([nD, nP], dtype=np.complex128)
    d = np.zeros([nD, 1], dtype=np.complex128)
    h = np.zeros([nD, 1], dtype=np.complex128)

    for i in range(nD):
        residues[i, :], d[i, :], h[i, :] = vector_fitting.calculate_residues(f[i, :].reshape(nF), s, poles, has_d=has_d, has_h=has_h)
    
    return residues, d, h


def mat2vec(f_mat):
    # Function to transform a symmetric matrix into a vector form
    # Input: [nD, nD, nF]
    # Output: [nD*(nD+1)/2, nF]
    nD = np.size(f_mat, 0)
    nF = np.size(f_mat, 2)
    
    f_vec = np.zeros([int(nD*(nD+1)/2), nF], dtype=np.complex128)
    idx = 0
    for i in range(nD):
        for j in range(i, nD):
            f_vec[idx, :] = f_mat[i, j, :]
            idx += 1
    return f_vec


def vec2mat(f_vec):
    # Function to transform a vector form into a symmetric matrix
    # Input: [nD*(nD+1)/2, nF]
    # Output: [nD, nD, nF]
    nD = np.size(f_vec, 0)
    nD = int((np.sqrt(nD*8+1)-1)/2) # calculate the dimension of the matrix
    nF = np.size(f_vec, 1)
    
    f_mat = np.zeros([nD, nD, nF], dtype=np.complex128)
    idx = 0
    for i in range(nD):
        for j in range(i, nD):
            f_mat[i, j, :] = f_vec[idx, :]
            if i != j:
                f_mat[j, i, :] = f_vec[idx, :]
            idx += 1
    if nF == 1:
        f_mat = f_mat.reshape([nD, nD])
    return f_mat


def matrix_fitting(f, s, n_poles=10, n_iters=10, has_d=1, has_h=1):
    # Function runs vector fitting
    # Assume w is imaginary, non-negative and in a ascending order
    w = np.imag(s)
    f_vec = mat2vec(f)
    poles = vector_fitting.init_poles(w[-1], n_poles)
    
    for loop in range(n_iters):
        poles = matrix_fitting_step(f_vec, s, poles, has_d=has_d, has_h=has_h)
    
    residues_vec, d_vec, h_vec = calculate_matrix_residues(f_vec, s, poles, has_d=has_d, has_h=has_h)
    residues = vec2mat(residues_vec)
    d = vec2mat(d_vec)
    h = vec2mat(h_vec)

    return poles, residues, d, h


def matrix_fitting_rescale(f, s, **kwargs):
    # Function rescales f and s and run vector fitting
    s_scale = abs(s[-1])
    f_scale = abs(f[0, 0, -1])
    
    poles_scaled, residues_scaled, d_scaled, h_scaled = matrix_fitting(f/f_scale, s/s_scale, **kwargs)
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
    tmp_list = [-3000,
                     -83000,
                     -5+7000j, -5-7000j,
                     -20+18000j, -20-18000j,
                     6000+45000j, 6000-45000j,
                     40+60000j, 40-60000j,
                     90+10000j, 90-10000j,
                     50000+80000j, 50000-80000j,
                     1000+45000j, 1000-45000j,
                     -5000+92000j, -5000-92000j]
    nD = 2
    nP = len(poles_test)
    nF = len(s_test)
    
    residues_test = np.zeros([nD, nD, nP], dtype=np.complex128)
    for i, r in enumerate(tmp_list):
        residues_test[:, :, i] = r*np.array([[1, 0.5], [0.5, 1]])
        #residues_test[:, :, i] = r*np.array([1])
    
    d_test = .2 * np.array([[1, -0.2], [-0.2, 1]])
    h_test = 5e-4 * np.array([[1, -0.5], [-0.5, 1]])
    
    # Input to model: nD dimension, nP poles, nF frequency,
    
    f_test = matrix_model(s_test, poles_test, residues_test, d_test, h_test)
    
    poles, residues, d, h = matrix_fitting(f_test, s_test, n_poles=18, has_d=0, has_h=1)
    #poles, residues, d, h = matrix_fitting_rescale(f_test, s_test, n_poles=10, n_iters=10)
    f_fit = matrix_model(s_test, poles, residues, d, h)
    
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_test[0, 1, :])), 'b-')
    plt.plot(np.abs(s_test), 20*np.log10(np.abs(f_fit[0, 1, :])), 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.show()


