# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vecfit


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
    s_50 = np.zeros(z.shape, dtype=z.dtype)
    cs = freq * 2j * np.pi
    # z0 = 50
    z0 = 190
    for i in range(len(freq)):
        s_50[:, :, i] = np.matrix(z[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(np.matrix(z[:, :, i] / z0 + np.identity(n)))

    # Fit even and odd mode separately
    s_even = ((s_50[0, 0, :] + s_50[0, 1, :] + s_50[1, 0, :] + s_50[1, 1, :]) / 2).reshape(len(freq))
    s_odd = ((s_50[0, 0, :] - s_50[0, 1, :] - s_50[1, 0, :] + s_50[1, 1, :]) / 2).reshape(len(freq))
    cs_all = np.logspace(10, 12, 1e5) * 1j

    # Even mode
    f_even = vecfit.fit_s(s_even, cs, n_pole=19, n_iter=20, s_inf=1, bound_wt=1.1)

    bound_even, bw_even = f_even.bound(np.inf, f0=39e9)
    print('Bound even is {:.5e}'.format(bound_even))
    print('BW even is {:.5e}'.format(bw_even))

    bound_error_even = f_even.bound_error(s_even, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_even))
    integral_even = f_even.bound_integral(cs, reflect=np.inf)
    print('The integral of the even is {:.5e}'.format(integral_even))

    print('check s for even', max(np.abs(f_even.model(cs_all))))

    # Odd mode
    f_odd = vecfit.fit_s(s_odd, cs, n_pole=17, n_iter=20, s_inf=1, bound_wt=0.6)

    bound_odd, bw_odd = f_odd.bound(np.inf, f0=39e9)
    print('Bound odd is {:.5e}'.format(bound_odd))
    print('BW odd is {:.5e}'.format(bw_odd))

    bound_error_odd = f_odd.bound_error(s_odd, cs, reflect=np.inf)
    print('Bound error is {:.5e}'.format(bound_error_odd))
    integral_odd = f_odd.bound_integral(cs, reflect=np.inf)
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


if __name__ == '__main__':
    # example1()
    # single_siw()
    coupled_siw()


