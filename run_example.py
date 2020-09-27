# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vecfit
import scipy.io
from os import makedirs
from os.path import expanduser, exists
import cmath
from scipy.interpolate import interp1d
from scipy import optimize


def plot_save(file_name=None, fig=None, format='pdf'):
    '''
    Save the figures to file
    :param file_name: Save file name.  If file name is None, save all current figures to default folder
    :param fig: Figure to save
    :return: None
    '''
    dir_name = expanduser('~/Documents/plots')
    if not exists(dir_name):
        makedirs(dir_name)

    # save the figures
    if fig is None:
        for i in plt.get_fignums():
            plt.figure(i)
            plt.savefig('{}/Figure_{}.{}'.format(dir_name, i, format), dpi=360, format=format)
            plt.close()
        # fig = plt
    else:
        fig.savefig(file_name, dpi=360, format=format)


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
    ax = f_in.plot(s_test, x_scale='log', y_scale='db')
    f_out.plot(s_test, ax=ax, y_scale='db', linestyle='--')
    (f_out-f_in).plot(s_test, ax=ax, y_scale='db', linestyle='--')
    ax.legend(['Input model', 'Output model', 'Error'])


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

    axs = vecfit.plot_freq_resp_matrix(s_test, f_test, y_scale='db')
    f_out.plot(s_test, axs, y_scale='db', linestyle='--')
    (f_out-f_in).plot(s_test, axs, y_scale='db', linestyle='--')


def single_siw():
    s1p_file = './resource/single_SIW_antenna_39GHz_50mil.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    z0 = 190
    s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=18, n_iter=20, s_dc=0, s_inf=1, bound_wt=0.27)
    f_out = vecfit.bound_tightening(s_data, cs)

    bound, bw = f_out.bound(np.inf, f0=39e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, x_scale='log', y_scale='db')
    f_out.plot(cs, ax=ax1, x_scale='log', y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, x_scale='log', y_scale='db', linestyle='--')
    ax2 = f_out.plot_improved_bound(1.2e10, 4e11)


def coupled_siw_even_odd():
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

    # Even mode
    # f_even = vecfit.fit_s(s_even, cs, n_pole=19, n_iter=20, s_inf=1, bound_wt=1.1)
    f_even = vecfit.bound_tightening(s_even, cs)
    bound_even, bw_even = f_even.bound(np.inf, f0=39e9)
    bound_error_even = f_even.bound_error(s_even, cs, reflect=np.inf)
    # integral_even = f_even.bound_integral(cs, reflect=np.inf)
    integral_even = vecfit.bound_integral(s_even, cs, np.inf)

    # Odd mode
    # f_odd = vecfit.fit_s(s_odd, cs, n_pole=17, n_iter=20, s_inf=1, bound_wt=0.62)
    f_odd = vecfit.bound_tightening(s_odd, cs)
    bound_odd, bw_odd = f_odd.bound(np.inf, f0=39e9)
    bound_error_odd = f_odd.bound_error(s_odd, cs, reflect=np.inf)
    # integral_odd = f_odd.bound_integral(cs, reflect=np.inf)
    integral_odd = vecfit.bound_integral(s_odd, cs, np.inf)

    print('Even mode: bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound_even, bw_even, bound_error_even, integral_even))
    print('Odd mode: bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound_odd, bw_odd, bound_error_odd, integral_odd))

    # plots
    ax1 = vecfit.plot_freq_resp(cs, s_even, y_scale='db')
    f_even.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_even-f_even.model(cs), ax=ax1, y_scale='db', linestyle='--')
    ax2 = f_even.plot_improved_bound(2e10, 4e11)

    ax3 = vecfit.plot_freq_resp(cs, s_odd, y_scale='db')
    f_odd.plot(cs, ax=ax3, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_odd-f_odd.model(cs), ax=ax3, y_scale='db', linestyle='--')
    ax4 = f_odd.plot_improved_bound(2e10, 4e11)


def coupled_siw():
    s2p_file = './resource/two_SIW_antenna_39GHz_50mil.s2p'
    freq, n, z, s, z0 = vecfit.read_snp(s2p_file)
    s_z0 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq * 2j * np.pi
    # z0 = 50
    z0 = 190
    for i in range(len(freq)):
        s_z0[:, :, i] = np.matrix(z[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(np.matrix(z[:, :, i] / z0 + np.identity(n)))

    s_model, bound = vecfit.mode_fitting(s_z0, cs, True)
    bound_error = s_model.bound_error(s_z0, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_z0, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))

    # plots
    axs = vecfit.plot_freq_resp_matrix(cs, s_z0, y_scale='db')
    s_model.plot(cs, axs=axs, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp_matrix(cs, s_z0-s_model.model(cs), axs=axs, y_scale='db', linestyle='--')


def transmission_line_model():
    # load: r // c + l
    r = 55
    l = 5e-9
    c = 1e-12

    # z0 = sqrt(l0 / c0) = 50 Ohm
    z0 = 50
    c0 = 1e-12
    l0 = 2.5e-9
    n = 9  # number of stages

    freq = np.linspace(1e9, 5e9, 1000)
    cs = freq*2j*np.pi
    zl = 1 / (c * cs + 1/r) + l * cs
    for i in range(n):
        zl = zl + l0 * cs
        zl = 1 / (c0 * cs + 1/zl)
    sl = (zl - z0) / (zl + z0)
    s_data = sl

    # Try to fit S
    f_out = vecfit.fit_s(s_data, cs, n_pole=19, n_iter=20, s_inf=-1)
    # f_out = vecfit.bound_tightening(s_data, cs)

    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')
    ax2 = f_out.plot_improved_bound(1e11, 4e10)


def dipole():
    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=3, n_iter=20, s_inf=-1)
    # f_out = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=-1, bound_wt=0.3)
    f_out = vecfit.bound_tightening(s_data, cs)

    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')
    ax2 = f_out.plot_improved_bound(1e11, 4e10)


def dipole_paper():
    # Equivalent circuit of a dipole antenna using frequency-independent lumped elements
    # https://ieeexplore.ieee.org/document/210122

    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi
    z0 = 50

    # [2] three-element
    # [3] four-element
    # proposed four-element: C1 + (C2 // L1 // R1)
    h = 62.5e-3/2
    a = 2.5e-3/2
    # h = 0.9
    # a = 0.00264
    C1 = 12.0674 * h / (np.log10(2 * h / a) - 0.7245) * 1e-12 # pF
    C2 = 2 * h * (0.89075 / (np.power(np.log10(2 * h / a), 0.8006) - 0.861) - 0.02541) * 1e-12 # pF
    L1 = 0.2 * h * (np.power(1.4813 * np.log10(2 * h / a), 1.012) - 0.6188) * 1e-6 # uH
    R1 = (0.41288 * np.power(np.log10(2 * h / a), 2) + 7.40754 * np.power(2 * h / a, -0.02389) - 7.27408) * 1e3 # k Ohm

    C1 *= z0
    C2 *= z0
    L1 /= z0
    R1 /= z0
    p1 = (-1/R1/C2 + cmath.sqrt(1/R1**2/C2**2 - 4/L1/C2)) / 2
    p2 = (-1/R1/C2 - cmath.sqrt(1/R1**2/C2**2 - 4/L1/C2)) / 2
    r1 = 1/C2 / (1 - p2/p1)
    r2 = 1/C2 / (1 - p1/p2)
    pole_model = [0, p1, p2]
    zero_model = [1 / C1, r1, r2]
    z_model = vecfit.RationalFct(pole_model, zero_model, 0, 0)
    s_model = (z_model.model(cs) - 1) / (z_model.model(cs) + 1)

    # Try to fit S
    f_out = vecfit.bound_tightening(s_data, cs)

    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_model, ax=ax1, y_scale='db')
    vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_data-s_model, ax=ax1, y_scale='db', linestyle='--')
    ax2 = f_out.plot_improved_bound(1e11, 4e10)


def dipole_bound_vs_pole():
    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.bound_tightening(s_data, cs)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 20):
        f_out = vecfit.fit_s(s_data, cs, n_pole=i, n_iter=20, s_inf=-1)
        bound, bw = f_out.bound(np.inf, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
        vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')

    # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.plot(pole_list, bound_list, '--')
    ax.plot(pole_list, bound_error_list, '--')
    ax.set_xlabel(r'$N_p$')
    ax.legend([r'$B+\delta B$', r'$B$', r'$\delta B$'])
    ax.grid(True, which='both', linestyle='--')
    # fig.savefig('/Users/dingnie/Desktop/dipole_bound_vs_poles.pdf', dpi=360, format='pdf')
    # plt.show()

    ax2 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 20):
        f_out = vecfit.fit_s(s_data, cs, n_pole=i, n_iter=20, s_dc=1)
        bound, bw = f_out.bound(0, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=0)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        f_out.plot(cs, ax=ax2, y_scale='db', linestyle='--')
        vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax2, y_scale='db', linestyle='--')

    # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.plot(pole_list, bound_list, '--')
    ax.plot(pole_list, bound_error_list, '--')
    plt.show()

    # # plot bound vs pole when th tightening method is used
    # ax3 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    # pole_list = []
    # bound_list = []
    # bound_error_list = []
    # for i in range(1, 20):
    #     f_out = vecfit.bound_tightening(s_data, cs, i)
    #     if f_out is None:
    #         bound = np.nan
    #         bound_error = np.nan
    #     else:
    #         bound, bw = f_out.bound(0, f0=2.4e9)
    #         bound_error = f_out.bound_error(s_data, cs, reflect=0)
    #     pole_list.append(i)
    #     bound_list.append(bound)
    #     bound_error_list.append(bound_error)
    #     # f_out.plot(cs, ax=ax3, y_scale='db', linestyle='--')
    #     # vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax3, y_scale='db', linestyle='--')
    #
    # # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    # fig = plt.figure(figsize=(8, 5.5))
    # ax = fig.add_subplot(111)
    # ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    # ax.plot(pole_list, bound_list, '--')
    # ax.plot(pole_list, bound_error_list, '--')
    # plt.show()


def long_dipole_bound_vs_pole():
    s1p_file = './resource/single_dipole_wideband.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.bound_tightening(s_data, cs)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 15):
        f_out = vecfit.fit_s(s_data, cs, n_pole=i, n_iter=20, s_inf=-1)
        bound, bw = f_out.bound(np.inf, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
        vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')

    # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.semilogy(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.semilogy(pole_list, bound_list, '--')
    ax.semilogy(pole_list, bound_error_list, '--')
    # plt.show()

    ax2 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 15):
        f_out = vecfit.fit_s(s_data, cs, n_pole=i, n_iter=20, s_dc=1)
        bound, bw = f_out.bound(0, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=0)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        f_out.plot(cs, ax=ax2, y_scale='db', linestyle='--')
        vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax2, y_scale='db', linestyle='--')

    # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.semilogy(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.semilogy(pole_list, bound_list, '--')
    ax.semilogy(pole_list, bound_error_list, '--')
    plt.show()


def short_dipole():
    s1p_file = './resource/short_dipole_data.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi

    # Fit S in two ways
    z0 = 50  # 50 Ohm
    f0 = 2.4e9  # 2.4 GHz
    # Model 1: C // R
    s_model_1 = vecfit.bound_tightening(s_data, cs)
    b1 = s_model_1.bound(0)
    # z = r / (2*s - r - 2*p)
    z_model_1 = vecfit.RationalFct([(s_model_1.residue[0] + 2*s_model_1.pole[0]) / 2], [s_model_1.residue[0] / 2], 0, 0)
    C = 2 / (s_model_1.residue[0] * z0)
    R = s_model_1.residue[0] * z0 / (-s_model_1.residue[0] - 2*s_model_1.pole[0])
    print('Model 1: C // R, C = {:.5e}, R = {:.5e}'.format(C.real, R.real))
    # model_1_resp = (z_model_1.model(cs) - 1) / (z_model_1.model(cs) + 1)

    # Model 2: R1 + C // R2, approximately R1 + C
    s_model_2 = vecfit.fit_s(s_data, cs, n_pole=1, n_iter=20, s_dc=1)
    b2 = s_model_2.bound(0)
    # z = ((1+d)*s + r - (1+d)*p) / ((1-d)*s - r - (1-d)*p) = (a*s + b) / (c*s + d)
    a = 1 + s_model_2.const
    b = s_model_2.residue[0] - (1 + s_model_2.const) * s_model_2.pole[0]
    c = 1 - s_model_2.const
    d = -s_model_2.residue[0] - (1 - s_model_2.const) * s_model_2.pole[0]  # d is a small negative number, set to 0
    z_model_2 = vecfit.RationalFct([0], [(b - a*d/c) / c], a / c, 0)
    R1 = a / c * z0
    tmp = b - a*d/c
    C = c / (tmp * z0)
    R2 = tmp * z0 / d  # negative
    print('Model 2: R1 + C, R1 = {:.5e}, C = {:.5e}'.format(R1.real, C.real))
    # model_2_resp = (z_model_2.model(cs) - 1) / (z_model_2.model(cs) + 1)

    # Textbook model: R + C
    R = 20 * np.pi**2 * (1/20)**2
    C = np.pi / (4e-7 * np.pi * 3e8**2) * (3e8 / 2.4e9 / 20 / 2) / (np.log(25) - 1)
    z_model_3 = vecfit.RationalFct([0], [1 / C / z0], R / z0, 0)
    s_model_3 = 0
    model_3_resp = (z_model_3.model(cs) - 1) / (z_model_3.model(cs) + 1)
    print('Model 3: R + C, R = {:.5e}, C = {:.5e}'.format(R.real, C.real))

    # f_out = vecfit.fit_s(s_data, cs, n_pole=1, n_iter=20, s_dc=1)
    # f_out = vecfit.bound_tightening(s_data, cs)
    # bound, bw = f_out.bound(np.inf, f0=2.4e9)
    # bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    # ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    # print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    s_model_1.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    s_model_2.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    # s_model_3.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, model_3_resp, ax=ax1, y_scale='db', linestyle='--')

    # vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')
    # ax2 = f_out.plot_improved_bound(1e11, 4e10)
    ax1.legend(['Simulation', 'Auto fitting', 'Manual fitting', 'Textbook'])


def coupled_dipole():
    s2p_file = './resource/coupled_dipoles.s2p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s2p_file)
    cs = freq*2j*np.pi

    # Fit even and odd mode separately
    s_even = ((s_data[0, 0, :] + s_data[0, 1, :] + s_data[1, 0, :] + s_data[1, 1, :]) / 2).reshape(len(freq))
    s_odd = ((s_data[0, 0, :] - s_data[0, 1, :] - s_data[1, 0, :] + s_data[1, 1, :]) / 2).reshape(len(freq))
    # Even and odd mode
    # f_even = vecfit.fit_s(s_even, cs, n_pole=3, n_iter=20, s_inf=-1)
    # f_odd = vecfit.fit_s(s_odd, cs, n_pole=3, n_iter=20, s_inf=-1)
    f_even = vecfit.bound_tightening(s_even, cs)
    f_odd = vecfit.bound_tightening(s_odd, cs)

    fig = plt.figure(figsize=(8, 5.5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    vecfit.plot_freq_resp(cs, s_even, ax=ax1, y_scale='db')
    f_even.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_odd, ax=ax2, y_scale='db')
    f_odd.plot(cs, ax=ax2, y_scale='db', linestyle='--')

    # Try to fit S
    # fixed_pole = np.concatenate([f_even.pole, f_odd.pole])
    # s_model = vecfit.matrix_fitting_rescale(s_data, cs, n_pole=6, n_iter=20, has_const=True, has_linear=False, fixed_pole=fixed_pole)
    # s_model = s_model.rank_one()
    # s_model = vecfit.matrix_fitting_rank_one_rescale(s_data, cs, n_pole=6, n_iter=50, has_const=True, has_linear=True)
    s_model, bound = vecfit.mode_fitting(s_data, cs, True)
    bound_error = s_model.bound_error(s_data, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))

    axs = vecfit.plot_freq_resp_matrix(cs, s_data, y_scale='db')
    s_model.plot(cs, axs=axs, y_scale='db', linestyle='--')


def skycross_antennas():
    snp_file = './resource/Skycross_4_parallel_fine2.s4p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(snp_file)
    cs = freq*2j*np.pi

    s_matrix, bound = vecfit.mode_fitting(s_data, cs, True)
    bound_error = s_matrix.bound_error(s_data, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))


    # plot per mode response
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
    int_fct = vecfit.f_integral(cs.imag, 0) / 2 * np.log(1 + (1 - tau ** 2) / tau ** 2 * rho)
    delta_b = vecfit.num_integral(cs.imag, int_fct)
    print('Bound error is {:.2e}'.format(delta_b))

    int_fct = vecfit.f_integral(cs.imag, 0) * (1 - np.sum(sigma**2, 1) / n)
    unmatched_integral = vecfit.num_integral(cs.imag, int_fct)
    print('Unmatched integral is {:.2e}'.format(unmatched_integral))

    # plot the s_matrix
    axs = vecfit.plot_freq_resp_matrix(cs, s_data, y_scale='db')
    s_matrix.plot(cs, axs=axs, y_scale='db', linestyle='--')


def skycross_antennas_old():
    snp_file = './resource/Skycross_4_parallel_fine2.s4p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(snp_file)
    # z0 = 50
    # s_data = (z_data-z0) / (z_data+z0)
    cs = freq*2j*np.pi
    cs_all = np.logspace(0, 15, 10000)*2j*np.pi

    # Fit modes separately
    s_inf = 1
    inf_weight = 1e6
    s_data = np.concatenate([s_data, np.reshape(inf_weight * np.identity(n), [n, n, 1])], 2)
    u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_data)
    s_data = s_data[:, :, :-1]
    Lambda_A = Lambda_A[:, :, :-1]
    A_remain = A_remain[:, :, :-1]

    # poles_by_mode = [5, 7, 9, 6]
    poles_by_mode = [5, 8, 5, 7]  # s_inf = 1
    f_by_mode = []
    total_bound = 0.0
    for i in range(n):
        s_mode = Lambda_A[i, i, :]
        f_mode = vecfit.fit_s(s_mode * np.exp(1j*np.pi*0), cs, n_pole=poles_by_mode[i], n_iter=20, s_dc=s_inf)
        mode_bound = f_mode.bound(0)
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

    # # do another fit for the remaining errors
    # s_inf = -1
    # inf_weight = 1e6
    # s_remain = np.concatenate([A_remain, np.reshape(inf_weight * np.identity(n), [n, n, 1])], 2)
    # u_a, Lambda_A, vh_a, A_remain, err_norm, orig_norm = vecfit.joint_svd(s_remain)
    # s_remain = s_remain[:, :, :-1]
    # Lambda_A = Lambda_A[:, :, :-1]
    # A_remain = A_remain[:, :, :-1]
    #
    # poles_by_mode = [7, 11, 8, 9]
    # f_by_mode = []
    # total_bound = 0.0
    # for i in range(n):
    #     s_mode = Lambda_A[i, i, :]
    #     f_mode = vecfit.fit_s(s_mode * np.exp(1j*np.pi*0), cs, n_pole=poles_by_mode[i], n_iter=20, s_inf=s_inf)
    #     mode_bound = f_mode.bound(np.inf)
    #     print('Mode {} bound is {:.2e}'.format(i, mode_bound[0]))
    #     total_bound += mode_bound[0]
    #     f_by_mode.append(f_mode)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot(np.abs(cs) / 2 / np.pi, 20 * np.log10(np.abs(s_mode)), 'b-')
    #     ax.semilogx(np.abs(cs_all) / 2 / np.pi, 20 * np.log10(np.abs(f_mode.model(cs_all))), 'r--')
    #     ax.grid(True)
    #     ax.set_xlabel('Frequency (Hz)')
    #     ax.set_ylabel('S even Amplitude (dB)')
    # total_bound /= n
    # print('Total bound is {:.2e}'.format(total_bound))
    # # Put the modes back to matrix
    # pole = np.zeros([np.sum(poles_by_mode)], dtype=np.complex128)
    # residue = np.zeros([n, n, np.sum(poles_by_mode)], dtype=np.complex128)
    # const = np.zeros([n, n], dtype=np.complex128)
    # idx = 0
    # for i in range(n):
    #     tmp_matrix = np.zeros([n, n], dtype=np.complex128)
    #     tmp_matrix[i, i] = 1
    #     pole_range = np.array(range(idx, idx+poles_by_mode[i]))
    #
    #     pole[pole_range] = f_by_mode[i].pole
    #     for j in range(poles_by_mode[i]):
    #         residue_matrix = np.dot(np.dot(u_a, f_by_mode[i].residue[j]*tmp_matrix), vh_a)
    #         residue[:, :, pole_range[j]] = residue_matrix
    #     const += np.dot(np.dot(u_a, f_by_mode[i].const*tmp_matrix), vh_a)
    #     idx += poles_by_mode[i]
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
    int_fct = vecfit.f_integral(cs.imag, 0) / 2 * np.log(1 + (1 - tau ** 2) / tau ** 2 * rho)
    delta_b = vecfit.num_integral(cs.imag, int_fct)
    print('Bound error is {:.2e}'.format(delta_b))

    int_fct = vecfit.f_integral(cs.imag, 0) * (1 - np.sum(sigma**2, 1) / n)
    unmatched_integral = vecfit.num_integral(cs.imag, int_fct)
    print('Unmatched integral is {:.2e}'.format(unmatched_integral))

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


def two_ifa():
    snp_file = './resource/Orthogonal_IFA_Free_Space.s2p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(snp_file)
    cs = freq*2j*np.pi

    # Fit modes separately
    s_matrix, bound = vecfit.mode_fitting(s_data, cs, True)
    bound_error = s_matrix.bound_error(s_data, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))

    # plot the s_matrix
    axs = vecfit.plot_freq_resp_matrix(cs, s_data, y_scale='db')
    s_matrix.plot(cs, axs=axs, y_scale='db', linestyle='--')


def four_ifa():
    snp_file = './resource/4Port_IFA_Free_Space.s4p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(snp_file)
    cs = freq*2j*np.pi

    # Fit modes separately
    s_matrix, bound = vecfit.mode_fitting(s_data, cs, True)
    bound_error = s_matrix.bound_error(s_data, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))

    # plot the s_matrix
    axs = vecfit.plot_freq_resp_matrix(cs, s_data, y_scale='db')
    s_matrix.plot(cs, axs=axs, y_scale='db', linestyle='--')


if __name__ == '__main__':
    # example1()
    # example2()
    # single_siw()
    # coupled_siw_even_odd()
    # coupled_siw()
    transmission_line_model()
    # dipole()
    # dipole_bound_vs_pole()
    # long_dipole_paper()
    # long_dipole_bound_vs_pole()
    # short_dipole()
    # coupled_dipole()
    # skycross_antennas()
    # two_ifa()
    # four_ifa()

    plt.show()
    # plot_save()


