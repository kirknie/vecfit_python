# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vecfit
import scipy.io


def plot_matrix(x, y, y2=None):
    fig = plt.figure()
    ndim = np.size(y, 0)
    idx = 1
    for i in range(ndim):
        for j in range(ndim):
            plot_idx = ndim * 100 + ndim * 10 + idx
            ax = fig.add_subplot(plot_idx)
            ax.plot(x, 20 * np.log10(np.abs(y[i, j, :])), 'b-')
            if y2 is not None:
                ax.plot(x, 20 * np.log10(np.abs(y2[i, j, :])), 'r--')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('S{}{} Amplitude (dB)'.format(i+1, j+1))
            idx += 1


def example1():
    s_test = 1j * np.linspace(1, 1e5, 800)
    poles_test = [-4500,
                  -41000,
                  -100 + 5000j, -100 - 5000j,
                  -120 + 15000j, -120 - 15000j,
                  -3000 + 35000j, -3000 - 35000j,
                  -200 + 45000j, -200 - 45000j,
                  -1500 + 45000j, -1500 - 45000j,
                  -500 + 70000j, -500 - 70000j,
                  -1000 + 73000j, -1000 - 73000j,
                  -2000 + 90000j, -2000 - 90000j]
    residues_test = [-3000,
                     -83000,
                     -5 + 7000j, -5 - 7000j,
                     -20 + 18000j, -20 - 18000j,
                     6000 + 45000j, 6000 - 45000j,
                     40 + 60000j, 40 - 60000j,
                     90 + 10000j, 90 - 10000j,
                     50000 + 80000j, 50000 - 80000j,
                     1000 + 45000j, 1000 - 45000j,
                     -5000 + 92000j, -5000 - 92000j]
    d_test = .2
    h_test = 5e-4

    f_in = vecfit.RationalFct(poles_test, residues_test, d_test, h_test)
    f_test = f_in.model(s_test)

    # f_out = vecfit.vector_fitting_rescale(f_test, s_test, n_pole=18, n_iter=10, has_const=True, has_linear=True, fixed_pole=[-1000, -5+6000j, -5-6000j])
    # f_out = vecfit.vector_fitting_rescale(f_test, s_test, n_pole=18, n_iter=10, has_const=True, has_linear=True, reflect_z=1j*2e4)
    # f_out = vecfit.vector_fitting_rescale(f_test, s_test, n_pole=18, n_iter=10, has_const=True, has_linear=True)
    f_out = vecfit.fit_z(f_test, s_test, n_pole=16, n_iter=10, has_const=True, has_linear=True, reflect_z=1j*2e4)
    f_fit = f_out.model(s_test)

    s_test = 1j * np.linspace(100, 1e6, 8000)
    # print(f_out.model([1j*2e4]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    f_in.plot(s_test, ax=ax, x_scale='log', y_scale='db', color='b')
    f_out.plot(s_test, ax=ax, y_scale='db', color='r', linestyle='--')
    (f_out-f_in).plot(s_test, ax=ax, y_scale='db', color='k', linestyle='--')
    plt.show()


def example2():
    s_test = 1j * np.linspace(1, 1e5, 800)
    poles_test = [-4500,
                  -41000,
                  -100 + 5000j, -100 - 5000j,
                  -120 + 15000j, -120 - 15000j,
                  -3000 + 35000j, -3000 - 35000j,
                  -200 + 45000j, -200 - 45000j,
                  -1500 + 45000j, -1500 - 45000j,
                  -500 + 70000j, -500 - 70000j,
                  -1000 + 73000j, -1000 - 73000j,
                  -2000 + 90000j, -2000 - 90000j]
    tmp_list = [-3000,
                -83000,
                -5 + 7000j, -5 - 7000j,
                -20 + 18000j, -20 - 18000j,
                6000 + 45000j, 6000 - 45000j,
                40 + 60000j, 40 - 60000j,
                90 + 10000j, 90 - 10000j,
                50000 + 80000j, 50000 - 80000j,
                1000 + 45000j, 1000 - 45000j,
                -5000 + 92000j, -5000 - 92000j]
    ndim = 2
    n_pole = len(poles_test)
    ns = len(s_test)

    residues_test = np.zeros([ndim, ndim, n_pole], dtype=np.complex128)
    for i, r in enumerate(tmp_list):
        residues_test[:, :, i] = r * np.array([[1, 0.5], [0.5, 1]])

    d_test = .2 * np.array([[1, -0.2], [-0.2, 1]])
    h_test = 5e-4 * np.array([[1, -0.5], [-0.5, 1]])

    f_in = vecfit.RationalMtx(poles_test, residues_test, d_test, h_test)
    f_test = f_in.model(s_test)

    # f_out = vecfit.matrix_fitting(f_test, s_test, n_pole=18, n_iter=20, has_const=True, has_linear=True)
    f_out = vecfit.matrix_fitting_rescale(f_test, s_test, n_pole=18, n_iter=10, has_const=True, has_linear=True, fixed_pole=[-1000, -5+6000j, -5-6000j])
    # f_out = f_out.rank_one()
    f_fit = f_out.model(s_test)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # f_in.plot(s_test, ax=ax, x_scale='log', y_scale='db', color='b')
    # f_out.plot(s_test, ax=ax, y_scale='db', color='r', linestyle='--')
    # (f_out-f_in).plot(s_test, ax=ax, y_scale='db', color='k', linestyle='--')
    plt.plot(np.abs(s_test)/2/np.pi, 20 * np.log10(np.abs(f_test[0, 0, :])), 'b-')
    plt.plot(np.abs(s_test)/2/np.pi, 20 * np.log10(np.abs(f_fit[0, 0, :])), 'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.show()


def single_siw():
    s1p_file = './resource/single_SIW_antenna_39GHz_50mil.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)

    # freq = freq[500:]
    # s_data = s_data[500:]
    # z0_data = z0_data[500:]
    # z_data = z_data[500:]
    # z0 = 50
    z0 = 190
    s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi

    # Try to fit S
    f_out = vecfit.fit_s(s_data, cs, n_pole=18, n_iter=20, s_dc=0, s_inf=1, bound_wt=0.27)

    bound, bw = f_out.bound(np.inf, f0=39e9)
    print('Bound is {:.5e}'.format(bound))
    print('BW is {:.5e}'.format(bw))

    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error))

    ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('The integral of the antenna is {:.5e}'.format(ant_integral))

    cs_all = np.logspace(10, 12, 1e5) * 1j
    print('check s', max(np.abs(f_out.model(cs_all))))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freq, 20*np.log10(np.abs(s_data)), color='b')
    f_out.plot(cs, ax=ax1, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax1.plot(freq, 20*np.log10(np.abs(s_data-f_out.model(cs))), color='k', linestyle='--')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    f_out.plot_improved_bound(1.2e10, 4e11, ax=ax2)

    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    # f_out.plot(cs_all, ax=ax3, x_scale='log', y_scale='db', color='r', linestyle='--')

    plt.show()


def coupled_siw():
    s2p_file = './resource/two_SIW_antenna_39GHz_50mil.s2p'
    freq, n, z, s, z0 = vecfit.read_snp(s2p_file)
    s_z0 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq * 2j * np.pi
    # z0 = 50
    z0 = 190
    for i in range(len(freq)):
        s_z0[:, :, i] = np.matrix(z[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(np.matrix(z[:, :, i] / z0 + np.identity(n)))

    # Fit even and odd mode separately
    s_even = ((s_z0[0, 0, :] + s_z0[0, 1, :] + s_z0[1, 0, :] + s_z0[1, 1, :]) / 2).reshape(len(freq))
    s_odd = ((s_z0[0, 0, :] - s_z0[0, 1, :] - s_z0[1, 0, :] + s_z0[1, 1, :]) / 2).reshape(len(freq))
    cs_all = np.logspace(10, 12, 1e5) * 1j

    # Even mode
    f_even = vecfit.fit_s(s_even, cs, n_pole=19, n_iter=20, s_inf=1, bound_wt=1.1)

    bound_even, bw_even = f_even.bound(np.inf, f0=39e9)
    print('Bound even is {:.5e}'.format(bound_even))
    print('BW even is {:.5e}'.format(bw_even))

    bound_error_even = f_even.bound_error(s_even, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_even))
    integral_even = f_even.bound_integral(cs, reflect=np.inf)
    integral_even = vecfit.bound_integral(s_even, cs, np.inf)
    print('The integral of the even is {:.5e}'.format(integral_even))

    print('check s for even', max(np.abs(f_even.model(cs_all))))

    # Odd mode
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=17, n_iter=20, s_inf=1, bound_wt=0.62)

    bound_odd, bw_odd = f_odd.bound(np.inf, f0=39e9)
    print('Bound odd is {:.5e}'.format(bound_odd))
    print('BW odd is {:.5e}'.format(bw_odd))

    bound_error_odd = f_odd.bound_error(s_odd, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_odd))
    integral_odd = f_odd.bound_integral(cs, reflect=np.inf)
    integral_odd = vecfit.bound_integral(s_odd, cs, np.inf)
    print('The integral of the odd is {:.5e}'.format(integral_odd))

    print('check s for odd', max(np.abs(f_odd.model(cs_all))))

    # plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freq, 20 * np.log10(np.abs(s_even)), 'b-')
    f_even.plot(cs, ax=ax1, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax1.plot(freq, 20 * np.log10(np.abs(s_even - f_even.model(cs))), 'k--')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    f_even.plot_improved_bound(2e10, 4e11, ax=ax2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(freq, 20 * np.log10(np.abs(s_odd)), 'b-')
    f_odd.plot(cs, ax=ax3, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax3.plot(freq, 20 * np.log10(np.abs(s_odd - f_odd.model(cs))), 'k--')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    f_odd.plot_improved_bound(2e10, 4e11, ax=ax4)

    plt.show()


def coupled_siw_rank_one():
    s2p_file = './resource/two_SIW_antenna_39GHz_50mil.s2p'
    freq, n, z, s, z0 = vecfit.read_snp(s2p_file)
    s_z0 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq * 2j * np.pi
    # z0 = 50
    z0 = 190
    for i in range(len(freq)):
        s_z0[:, :, i] = np.matrix(z[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(np.matrix(z[:, :, i] / z0 + np.identity(n)))
    cs_all = np.logspace(10, 12, 1e5) * 1j
    
    # try the poles of even and odd mode
    s_even = ((s_z0[0, 0, :] + s_z0[0, 1, :] + s_z0[1, 0, :] + s_z0[1, 1, :]) / 2).reshape(len(freq))
    s_odd = ((s_z0[0, 0, :] - s_z0[0, 1, :] - s_z0[1, 0, :] + s_z0[1, 1, :]) / 2).reshape(len(freq))
    f_even = vecfit.fit_s(s_even, cs, n_pole=19, n_iter=20, s_inf=1, bound_wt=1.1)
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=17, n_iter=20, s_inf=1, bound_wt=0.62)
    poles = np.concatenate([f_even.pole, f_odd.pole])

    # f_out = vecfit.matrix_fitting_rescale(s_z0, cs, n_pole=36, n_iter=10, has_const=True, has_linear=True)
    # f_out = f_out.rank_one()
    f_out = vecfit.matrix_fitting_rank_one_rescale(s_z0, cs, n_pole=36, n_iter=10, has_const=True, has_linear=True, fixed_pole=None)
    f_fit = f_out.model(cs)

    plot_matrix(np.abs(cs)/2/np.pi, s_z0, f_fit)
    plt.show()


def dipole():
    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    # z0 = 50
    # s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=3, n_iter=20, s_inf=-1)
    # f_out = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=-1)
    f_out = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=-1, bound_wt=0.3)

    # f_out = vecfit.fit_s(s_data, cs, n_pole=9, n_iter=20, s_dc=1)

    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    print('Bound is {:.5e}'.format(bound))
    print('BW is {:.5e}'.format(bw))

    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error))

    ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('The integral of the antenna is {:.5e}'.format(ant_integral))

    cs_all = np.logspace(5, 12, 1e5) * 1j
    print('check s', max(np.abs(f_out.model(cs_all))))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freq, 20*np.log10(np.abs(s_data)), color='b')
    f_out.plot(cs, ax=ax1, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax1.plot(freq, 20*np.log10(np.abs(s_data-f_out.model(cs))), color='k', linestyle='--')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    f_out.plot_improved_bound(1e11, 4e10, ax=ax2)

    plt.show()


def coupled_dipole():
    s2p_file = './resource/coupled_dipoles.s2p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s2p_file)
    # z0 = 50
    # s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi

    # Fit even and odd mode separately
    s_even = ((s_data[0, 0, :] + s_data[0, 1, :] + s_data[1, 0, :] + s_data[1, 1, :]) / 2).reshape(len(freq))
    s_odd = ((s_data[0, 0, :] - s_data[0, 1, :] - s_data[1, 0, :] + s_data[1, 1, :]) / 2).reshape(len(freq))
    # Even and odd mode
    f_even = vecfit.fit_s(s_even, cs, n_pole=3, n_iter=20, s_inf=-1)
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=3, n_iter=20, s_inf=-1)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(s_even)), 'b-')
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(f_even.model(cs))), 'r--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('S even Amplitude (dB)')
    ax = fig.add_subplot(212)
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(s_odd)), 'b-')
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(f_odd.model(cs))), 'r--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('S odd Amplitude (dB)')
    # plt.show()

    # Try to fit S
    fixed_pole = np.concatenate([f_even.pole, f_odd.pole])
    # f_out = vecfit.matrix_fitting_rescale(s_data, cs, n_pole=6, n_iter=20, has_const=True, has_linear=False, fixed_pole=fixed_pole)
    # f_out = f_out.rank_one()
    f_out = vecfit.matrix_fitting_rank_one_rescale(s_data, cs, n_pole=6, n_iter=50, has_const=True, has_linear=True)
    f_fit = f_out.model(cs)

    plot_matrix(np.abs(cs)/2/np.pi, s_data, f_fit)
    plt.show()


def coupled_siw_joint_svd_test():
    s2p_file = './resource/two_SIW_antenna_39GHz_50mil.s2p'
    freq, n, z, s, z0 = vecfit.read_snp(s2p_file)
    s_z0 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq * 2j * np.pi
    # z0 = 50
    z0 = 190
    for i in range(len(freq)):
        s_z0[:, :, i] = np.matrix(z[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(
            np.matrix(z[:, :, i] / z0 + np.identity(n)))
    cs_all = np.logspace(10, 12, 1e5) * 1j
    u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_z0)
    s_odd = Lambda_A[0, 0, :]
    s_even = Lambda_A[1, 1, :]
    print('sum of original norm square is {:.5e}'.format(orig_norm))
    print('sum of error norm square is {:.5e}'.format(err_norm))
    ratio = err_norm/orig_norm
    print('ratio is {:.5e}'.format(ratio))
    # Even mode
    f_even = vecfit.fit_s(s_even, cs, n_pole=19, n_iter=20, s_inf=1, bound_wt=1.1)

    bound_even, bw_even = f_even.bound(np.inf, f0=39e9)
    print('Bound even is {:.5e}'.format(bound_even))
    print('BW even is {:.5e}'.format(bw_even))

    bound_error_even = f_even.bound_error(s_even, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_even))
    integral_even = f_even.bound_integral(cs, reflect=np.inf)
    integral_even = vecfit.bound_integral(s_even, cs, np.inf)
    print('The integral of the even is {:.5e}'.format(integral_even))

    print('check s for even', max(np.abs(f_even.model(cs_all))))

    # Odd mode
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=17, n_iter=20, s_inf=1, bound_wt=0.62)

    bound_odd, bw_odd = f_odd.bound(np.inf, f0=39e9)
    print('Bound odd is {:.5e}'.format(bound_odd))
    print('BW odd is {:.5e}'.format(bw_odd))

    bound_error_odd = f_odd.bound_error(s_odd, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_odd))
    integral_odd = f_odd.bound_integral(cs, reflect=np.inf)
    integral_odd = vecfit.bound_integral(s_odd, cs, np.inf)
    print('The integral of the odd is {:.5e}'.format(integral_odd))

    print('check s for odd', max(np.abs(f_odd.model(cs_all))))

    # plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(freq, 20 * np.log10(np.abs(s_even)), 'b-')
    f_even.plot(cs, ax=ax1, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax1.plot(freq, 20 * np.log10(np.abs(s_even - f_even.model(cs))), 'k--')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    f_even.plot_improved_bound(2e10, 4e11, ax=ax2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(freq, 20 * np.log10(np.abs(s_odd)), 'b-')
    f_odd.plot(cs, ax=ax3, x_scale='log', y_scale='db', color='r', linestyle='--')
    ax3.plot(freq, 20 * np.log10(np.abs(s_odd - f_odd.model(cs))), 'k--')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    f_odd.plot_improved_bound(2e10, 4e11, ax=ax4)

    plt.show()


def coupled_dipole_joint_svd_test():
    s2p_file = './resource/coupled_dipoles.s2p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s2p_file)
    # z0 = 50
    # s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi

    # Fit even and odd mode separately
    u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_data)
    s_even = Lambda_A[0, 0, :]
    s_odd = Lambda_A[1, 1, :]
    # Even and odd mode
    f_even = vecfit.fit_s(s_even, cs, n_pole=3, n_iter=20, s_inf=-1)
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=3, n_iter=20, s_inf=-1)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(s_even)), 'b-')
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(f_even.model(cs))), 'r--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('S even Amplitude (dB)')
    ax = fig.add_subplot(212)
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(s_odd)), 'b-')
    ax.plot(np.abs(cs)/2/np.pi, 20 * np.log10(np.abs(f_odd.model(cs))), 'r--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('S odd Amplitude (dB)')
    # plt.show()

    # Try to fit S
    fixed_pole = np.concatenate([f_even.pole, f_odd.pole])
    # f_out = vecfit.matrix_fitting_rescale(s_data, cs, n_pole=6, n_iter=20, has_const=True, has_linear=False, fixed_pole=fixed_pole)
    # f_out = f_out.rank_one()
    f_out = vecfit.matrix_fitting_rank_one_rescale(s_data, cs, n_pole=6, n_iter=50, has_const=True, has_linear=True)
    f_fit = f_out.model(cs)

    plot_matrix(np.abs(cs)/2/np.pi, s_data, f_fit)
    plt.show()


def skycross_antennas():
    snp_file = './resource/Skycross_4_parallel_fine2.s4p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(snp_file)
    # z0 = 50
    # s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi
    cs_all = np.logspace(0, 15, 10000)*2j*np.pi

    # Fit modes separately
    s_inf = -1
    inf_weight = 1e6
    s_data = np.concatenate([s_data, np.reshape(s_inf * inf_weight * np.identity(n), [n, n, 1])], 2)
    u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_data)
    s_data = s_data[:, :, :-1]
    Lambda_A = Lambda_A[:, :, :-1]
    A_remain = A_remain[:, :, :-1]

    # poles_by_mode = [5, 7, 9, 6]
    poles_by_mode = [5, 6, 5, 6]
    f_by_mode = []
    total_bound = 0.0
    for i in range(n):
        s_mode = Lambda_A[i, i, :]
        f_mode = vecfit.fit_s(s_mode * np.exp(1j*np.pi*0), cs, n_pole=poles_by_mode[i], n_iter=20, s_inf=s_inf)
        mode_bound = f_mode.bound(np.inf)
        print('Mode {} bound is {:.2e}'.format(i, mode_bound[0]))
        total_bound += mode_bound[0]
        f_by_mode.append(f_mode)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.abs(cs) / 2 / np.pi, 20 * np.log10(np.abs(s_mode)), 'b-')
        ax.semilogx(np.abs(cs_all) / 2 / np.pi, 20 * np.log10(np.abs(f_mode.model(cs_all))), 'r--')
        ax.grid(True)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('S even Amplitude (dB)')
    total_bound /= n
    print('Total bound is {:.2e}'.format(total_bound))
    # Put the modes back to matrix
    pole = np.zeros([np.sum(poles_by_mode)], dtype=np.complex128)
    residue = np.zeros([n, n, np.sum(poles_by_mode)], dtype=np.complex128)
    const = np.zeros([n, n], dtype=np.complex128)
    idx = 0
    for i in range(n):
        tmp_matrix = np.zeros([n, n], dtype=np.complex128)
        tmp_matrix[i, i] = 1
        pole_range = np.array(range(idx, idx+poles_by_mode[i]))

        pole[pole_range] = f_by_mode[i].pole
        for j in range(poles_by_mode[i]):
            residue_matrix = np.dot(np.dot(u_a, f_by_mode[i].residue[j]*tmp_matrix), vh_a)
            residue[:, :, pole_range[j]] = residue_matrix
        const += np.dot(np.dot(u_a, f_by_mode[i].const*tmp_matrix), vh_a)
        idx += poles_by_mode[i]
    s_matrix = vecfit.RationalMtx(pole, residue, const)

    # do another fit for the remaining errors
    s_inf = -1
    inf_weight = 1e6
    s_remain = np.concatenate([A_remain, np.reshape(s_inf * inf_weight * np.identity(n), [n, n, 1])], 2)
    u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_remain)
    s_remain = s_remain[:, :, :-1]
    Lambda_A = Lambda_A[:, :, :-1]
    A_remain = A_remain[:, :, :-1]

    poles_by_mode = [7, 11, 8, 9]
    f_by_mode = []
    total_bound = 0.0
    for i in range(n):
        s_mode = Lambda_A[i, i, :]
        f_mode = vecfit.fit_s(s_mode * np.exp(1j*np.pi*0), cs, n_pole=poles_by_mode[i], n_iter=20, s_inf=s_inf)
        mode_bound = f_mode.bound(np.inf)
        print('Mode {} bound is {:.2e}'.format(i, mode_bound[0]))
        total_bound += mode_bound[0]
        f_by_mode.append(f_mode)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.abs(cs) / 2 / np.pi, 20 * np.log10(np.abs(s_mode)), 'b-')
        ax.semilogx(np.abs(cs_all) / 2 / np.pi, 20 * np.log10(np.abs(f_mode.model(cs_all))), 'r--')
        ax.grid(True)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('S even Amplitude (dB)')
    total_bound /= n
    print('Total bound is {:.2e}'.format(total_bound))
    # Put the modes back to matrix
    pole = np.zeros([np.sum(poles_by_mode)], dtype=np.complex128)
    residue = np.zeros([n, n, np.sum(poles_by_mode)], dtype=np.complex128)
    const = np.zeros([n, n], dtype=np.complex128)
    idx = 0
    for i in range(n):
        tmp_matrix = np.zeros([n, n], dtype=np.complex128)
        tmp_matrix[i, i] = 1
        pole_range = np.array(range(idx, idx+poles_by_mode[i]))

        pole[pole_range] = f_by_mode[i].pole
        for j in range(poles_by_mode[i]):
            residue_matrix = np.dot(np.dot(u_a, f_by_mode[i].residue[j]*tmp_matrix), vh_a)
            residue[:, :, pole_range[j]] = residue_matrix
        const += np.dot(np.dot(u_a, f_by_mode[i].const*tmp_matrix), vh_a)
        idx += poles_by_mode[i]
    # s_matrix = vecfit.RationalMtx(pole, residue, const) + s_matrix

    # calculate the bound error
    s_fit = s_matrix.model(cs)
    s_error = s_fit - s_data

    u, sigma, vh = np.linalg.svd(np.moveaxis(s_error, -1, 0))
    sigma_error = sigma[:, 0].flatten()
    u, sigma, vh = np.linalg.svd(np.moveaxis(s_data, -1, 0))
    sigma_max = sigma[:, 0].flatten()
    sigma_min = sigma[:, -1].flatten()

    rho = (2 * sigma_error) / (1 - np.power(sigma_max, 2)) * np.sqrt(1 + (sigma_max**2 - sigma_min**2) / (1 - sigma_max)**2)


    tau = 0.3162278
    int_fct = vecfit.f_integral(cs.imag, np.inf) / 2 * np.log(1 + (1 - tau ** 2) / tau ** 2 * rho)
    delta_b = vecfit.num_integral(cs.imag, int_fct)
    print('Bound error is {:.2e}'.format(delta_b))

    # plot the s_matrix
    s = 2j * np.pi * freq
    s_model = s_matrix.model(s)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(321)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[0, 0, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[0, 0, :].flatten())), 'r--')
    # ax1.plot(freq, 20 * np.log10(np.abs((s_data - s_model)[0, 0, :].flatten())), 'k--')

    ax1 = fig1.add_subplot(322)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[0, 1, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[0, 1, :].flatten())), 'r--')

    ax1 = fig1.add_subplot(323)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[0, 2, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[0, 2, :].flatten())), 'r--')

    ax1 = fig1.add_subplot(324)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[0, 3, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[0, 3, :].flatten())), 'r--')

    ax1 = fig1.add_subplot(325)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[1, 1, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[1, 1, :].flatten())), 'r--')

    ax1 = fig1.add_subplot(326)
    ax1.plot(freq, 20 * np.log10(np.abs(s_model[1, 2, :].flatten())), 'b-')
    ax1.plot(freq, 20 * np.log10(np.abs(s_data[1, 2, :].flatten())), 'r--')

    plt.show()


if __name__ == '__main__':
    # example1()
    # example2()
    # single_siw()
    # coupled_siw_joint_svd_test()
    # coupled_dipole_joint_svd_test()
    # coupled_siw_rank_one()
    # dipole()
    # coupled_dipole()
    skycross_antennas()


