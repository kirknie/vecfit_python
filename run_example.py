# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import vecfit
import scipy.io
from os import makedirs
from os.path import expanduser, exists
import cmath
from scipy.interpolate import interp1d
from scipy import optimize
import copy


default_figure_size = (8, 5.5)

# Thin lines
thin_linewidth = 0.5
thin_markersize = 3
# Medium lines
medium_linewidth = 1
medium_markersize = 5
# Thick lines
thick_linewidth = 1.5
thick_markersize = 5

default_linewidth = medium_linewidth
default_markersize = medium_markersize
default_barwidth = 0.8
default_alpha = 0.3
default_grid_linewidth = 0.5
default_grid_style = '--'
default_grid_color = '#b0b0b0'
default_font_size = 12
default_font_size_major = 'medium'
# number or {'xx-small', 'x-small', 'smaller', 'small', 'medium', 'large', 'larger', 'x-large', 'xx-large'}
default_tick_direction = 'in'

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau10 = tableau20[0::2]
tableau10_light = tableau20[1::2]

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = default_font_size
if 'SFHello-Light' in matplotlib.rcParams['font.family'] or 'SFHello-Regular' in matplotlib.rcParams['font.family']:
    # This is required to embed SF Hello fonts to pdf, the pdf size will be big though
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
# we may use a larger font size for some major information
matplotlib.rcParams['figure.titlesize'] = default_font_size_major
matplotlib.rcParams['axes.titlesize'] = default_font_size_major
matplotlib.rcParams['axes.labelsize'] = default_font_size_major
matplotlib.rcParams['xtick.labelsize'] = default_font_size_major
matplotlib.rcParams['ytick.labelsize'] = default_font_size_major
matplotlib.rcParams['xtick.direction'] = default_tick_direction
matplotlib.rcParams['ytick.direction'] = default_tick_direction
matplotlib.rcParams['grid.color'] = default_grid_color
matplotlib.rcParams['grid.linestyle'] = default_grid_style
matplotlib.rcParams['grid.linewidth'] = default_grid_linewidth
matplotlib.rcParams['lines.linewidth'] = default_linewidth
matplotlib.rcParams['patch.linewidth'] = default_grid_linewidth  # use this to control the linewidth of smith chart

def colors(i, j=None):
    all_color = tableau10 + tableau10_light
    all_color = [(r/255, g/255, b/255) for r, g, b in all_color]

    if isinstance(i, list):
        return [colors(idx) for idx in i]

    i = i % len(all_color)
    if j is not None:
        # return a list of j colors
        if j <= len(all_color[i:]):
            return all_color[i:i+j]
        else:
            return_color = all_color[i:]
            while j - len(return_color) > len(all_color):
                return_color = return_color + all_color
            return_color = return_color + all_color[:j-len(return_color)]
            return return_color
    else:
        return all_color[i]


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
    r = 60
    l = 5e-9
    c = 1e-12

    # z0 = sqrt(l0 / c0) = 50 Ohm
    z0 = 50
    c0 = 1e-12
    l0 = 2.5e-9
    n = 9  # number of stages

    freq = np.linspace(1e8, 3e9, 1000)
    cs = freq*2j*np.pi
    zl = 1 / (c * cs + 1/r) + l * cs
    for i in range(n):
        zl = zl + l0 * cs
        zl = 1 / (c0 * cs + 1/zl)
    sl = (zl - z0) / (zl + z0)
    s_data = sl

    # Try to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=19, n_iter=20, s_inf=-1)
    # f_out = vecfit.bound_tightening(s_data, cs)
    f_out, log = vecfit.bound_tightening_sweep(s_data, cs, np.inf)

    # try to draw a model for the transmission line model
    z_fit = (1 + f_out.model(cs)) / (1 - f_out.model(cs)) * z0
    z_fit_model = vecfit.fit_z(z_fit, cs, n_pole=7, n_iter=20, has_const=False, has_linear=False)
    axz = vecfit.plot_freq_resp(cs, z_fit, y_scale='db')
    z_fit_model.plot(cs, ax=axz, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, z_fit-z_fit_model.model(cs), y_scale='db')

    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    # ant_integral = f_out.bound_integral(cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(s_data, cs, np.inf)
    print('Model has {} poles'.format(f_out.pole.shape[0]))
    print('Bound is {:.5e}, BW is {:.5e}, Bound error is {:.5e}, sum is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bw, bound_error, bound + bound_error, ant_integral))

    ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
    vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')
    ax2 = f_out.plot_improved_bound(1e11, 4e10)


def transmission_line_model_vs_freq_range():
    # load: r // c + l
    r = 60
    l = 5e-9
    c = 1e-12

    # z0 = sqrt(l0 / c0) = 50 Ohm
    z0 = 50
    c0 = 1e-12
    l0 = 2.5e-9
    n = 9  # number of stages

    # start freq 0.1 GHz, end freq 3/4/5 GHz
    f1 = 1e8
    f2_list = [3e9, 4e9, 5e9, 6e9]
    bound_list = []
    bound_error_list = []
    f_out_list = []
    pole_num_list = []
    for f2 in f2_list:
        freq = np.linspace(f1, f2, 1000)
        cs = freq*2j*np.pi
        zl = 1 / (c * cs + 1/r) + l * cs
        for i in range(n):
            zl = zl + l0 * cs
            zl = 1 / (c0 * cs + 1 / zl)
        sl = (zl - z0) / (zl + z0)
        s_data = sl

        # Try to fit S
        # f_out = vecfit.fit_s(s_data, cs, n_pole=19, n_iter=20, s_inf=-1)
        # f_out = vecfit.bound_tightening(s_data, cs)
        f_out, log = vecfit.bound_tightening_sweep(s_data, cs, np.inf)

        bound, bw = f_out.bound(np.inf, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)

        f_out_list.append(f_out)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        pole_num_list.append(f_out.pole.shape[0])
        print('Poles {}, bound {:.5e}, error {:.5e}, sum {:.5e}'.format(len(f_out.pole), bound, bound_error, bound + bound_error))

    f_out = vecfit.fit_s(s_data, cs, n_pole=19, n_iter=20, s_inf=-1)
    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
    print('Poles {}, bound {:.5e}, error {:.5e}, sum {:.5e}'.format(len(f_out.pole), bound, bound_error,
                                                                    bound + bound_error))

    fig = plt.figure(figsize=(8, 5.5))
    ax1 = fig.add_subplot(111)
    vecfit.plot_freq_resp(cs, s_data, ax=ax1, y_scale='db', color=colors(0))
    i = 1
    for f_out, f2 in zip(f_out_list, f2_list):
        f_band = copy.copy(freq)
        f_band[freq > f2] = np.nan
        cs = f_band*2j*np.pi
        f_out.plot(cs, ax=ax1, y_scale='db', color=colors(i))
        vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--', color=colors(i))
        i += 1
    legend_text = ['S-param data']
    for i, pole_num in enumerate(pole_num_list):
        legend_text += ['Model {} ({} poles)'.format(i+1, pole_num), 'Model {} error'.format(i+1)]
    ax1.legend(legend_text)

    # extended frequency response
    fig = plt.figure(figsize=(8, 5.5))
    ax1 = fig.add_subplot(111)
    vecfit.plot_freq_resp(cs, s_data, ax=ax1, y_scale='db', color=colors(0))
    i = 1
    for f_out, f2 in zip(f_out_list, f2_list):
        f_band = copy.copy(freq)
        f_band[freq > f2] = np.nan
        cs = f_band*2j*np.pi
        vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', color=colors(i))
        i += 1
    legend_text = ['S-param data']
    for i, pole_num in enumerate(pole_num_list):
        legend_text += ['Error for Model {} ({} poles)'.format(i+1, pole_num)]
    ax1.legend(legend_text)
    i = 1
    for f_out, f2 in zip(f_out_list, f2_list):
        f_band = copy.copy(freq)
        f_band[freq < f2] = np.nan
        cs = f_band*2j*np.pi
        vecfit.plot_freq_resp(cs, s_data-f_out.model(cs), ax=ax1, y_scale='db', linestyle='--', color=colors(i))
        i += 1

    # pole vs bound
    fig = plt.figure(figsize=(8, 5.5))
    ax2 = fig.add_subplot(111)
    ax2.plot(pole_num_list, np.array(bound_list) + np.array(bound_error_list))
    ax2.plot(pole_num_list, bound_list, '--')
    ax2.plot(pole_num_list, bound_error_list, '--')
    ax2.legend([r'$B+\delta B$', r'$B$', r'$\delta B$'])


def model_brune_synthesis():
    from vecfit.rational_fct import RationalFct
    pole = [-3.73390786, -1.28424571-8.71367424j, -1.28424571+8.71367424j,
            -0.436489866-14.8638563j, -0.436489866+14.8638563j,
            -0.0829360526-21.3625685j, -0.0829360526+21.3625685j]
    residue = [186.067925, 93.0835654+26.4856515j, 93.0835654-26.4856515j,
               88.9730245+8.27914755j, 88.9730245-8.27914755j,
               132.304907+5.31930782j, 132.304907-5.31930782j]
    z_model = RationalFct(pole, residue)

    # Brune synthesis
    import sympy
    # 1. remove zero at infinity
    s = sympy.symbols('s')
    z = 0
    for p, r in zip(pole, residue):
        z += r / (s - p)
    # print(z)
    # print(sympy.simplify(z))
    z_pz = sympy.simplify(z)
    n, d = sympy.fraction(z_pz)
    # print(n)
    # print(d)
    n = 814.7909188*s**6 + 5883.08552836939*s**5 + 455058.612437942*s**4 + 2735360.9352182*s**3 + 63655976.6704326*s**2 + 254736247.151332*s + 1779434631.82901
    d = 1.0*s**7 + 7.3412511172*s**6 + 771.350936978897*s**5 + 5086.02945258848*s**4 + 163024.909675002*s**3 + 870212.844549592*s**2 + 8922478.53021907*s + 29231404.4495841
    # shunt C0 for (1/814.7909188) e-9 F
    C0 = 1/814.7909188
    d1 = sympy.simplify(d - C0 * s * n)
    # print(d1)
    z1 = n / d1

    # 2. find the frequency where real part of z1 is minimum
    all_freq = np.linspace(0, 40, 10000)
    all_freq = np.linspace(27.430, 27.435, 10000)
    z1_f = sympy.lambdify(s, z1, "numpy")
    z1_num = z1_f(1j*all_freq)
    idx = np.argmin(z1_num.real)
    w1 = all_freq[idx]
    r1 = z1_num.real[idx]
    x1 = z1_num.imag[idx]
    # plt.plot(all_freq, z1_num.real)
    # plt.grid(True)
    # minimum between 27.430 and 27.435
    # print(w1, r1, x1)
    z2 = z1 - r1
    # x1 > 0, case B
    y2 = sympy.simplify(1/z2)
    y2_f = sympy.lambdify(s, y2, "numpy")
    y2_num = y2_f(1j*w1).imag
    # print(y2_num, y2_num/w1)
    # take out shunt negative capacitance C1 y2_num / w1 = -0.00046727390293072437 e-9
    # also take out the zero at w1 = 27.43233223322332
    C1 = y2_num/w1
    n3, d3 = sympy.fraction(sympy.simplify(y2 - s * C1))
    # print(n3)
    # print(d3)
    n4 = sympy.apart(n3 / (s**2 + w1**2))
    n4 = 0.380730360161106*s**5 + 2.8695974853059*s**4 + 138.975976252091*s**3 + 847.472351426199*s**2 + 10059.3675296087*s + 38844.0243923608
    # # do a manual partial fraction for d3 / n3 here...
    # print(d3)
    # a1, a2, a3, a4, a5, a6 = sympy.symbols('a1 a2 a3 a4 a5 a6')
    # print(sympy.expand(n4 * s * a1 + (a2 * s**4 + a3 * s**3 + a4 * s**2 + a5 * s + a6) * (s**2 + w1**2)))
    # tmp_expr = sympy.collect(sympy.expand(n4 * s * a1 + (a2 * s**4 + a3 * s**3 + a4 * s**2 + a5 * s + a6) * (s**2 + w1**2) - d3), s)
    # print(tmp_expr)
    # print(tmp_expr.coeff(s, 6))
    # solve AA * xx = bb
    bb = [814.790549553869, 5882.43538336516, 455053.331652566, 2735101.61650645, 63654273.6064438, 254715664.721133, 1779345346.7171]
    AA = [[0.380730360161106, 1, 0, 0, 0, 0],
          [2.8695974853059, 0, 1, 0, 0, 0],
          [138.975976252091, 752.532851753943, 0, 1, 0, 0],
          [847.472351426199, 0, 752.532851753943, 0, 1, 0],
          [10059.3675296087, 0, 0, 752.532851753943, 0, 1],
          [38844.0243923608, 0, 0, 0, 752.532851753943, 0],
          [0, 0, 0, 0, 0, 752.532851753943]]
    xx, *other = np.linalg.lstsq(AA, bb)
    # a1 = xx[0] for the shunt LC circuit
    C3 = 1/xx[0]
    L3 = xx[0]/(w1**2)
    d4 = xx[1] * s ** 4 + xx[2] * s ** 3 + xx[3] * s ** 2 + xx[4] * s + xx[5]
    # print(d4)
    z4 = d4 / n4
    # shunt C4 for (0.380730360161106/247.97122460336) e-9 F
    C4 = (0.380730360161106/247.97122460336)
    d5 = sympy.simplify(n4 - C4 * s * d4)
    d5 = 0.397220177872962*s**4 + 44.4827847829576*s**3 + 445.769056411886*s**2 + 6428.99655736558*s + 38844.0243923608
    # print(d5)
    z5 = d4 / d5

    # 3. rinse and repeat
    all_freq = np.linspace(0, 40, 10000)
    all_freq = np.linspace(17.164, 17.165, 10000)
    z5_f = sympy.lambdify(s, z5, "numpy")
    z5_num = z5_f(1j * all_freq)
    idx = np.argmin(z1_num.real)
    w5 = all_freq[idx]
    r5 = z5_num.real[idx]
    x5 = z5_num.imag[idx]
    # plt.plot(all_freq, z5_num.real)
    # plt.grid(True)
    # minimum between 17.162 and 17.168
    # print(w5, r5, x5)
    z6 = z5 - r5
    # x5 > 0, case B
    y6 = sympy.simplify(1/z6)
    y6_f = sympy.lambdify(s, y6, "numpy")
    y6_num = y6_f(1j*w5).imag
    # print(y6_num, y6_num/w5)
    # take out shunt negative capacitance C6 y6_num / w5 = -0.0010925270115166507 e-9
    # also take out the zero at w5 = 17.164466446644663
    C6 = y6_num/w5
    n6, d6 = sympy.fraction(sympy.simplify(y6 - s * C6))
    # print(n6)
    # print(d6)
    n7 = sympy.apart(n6 / (s ** 2 + w5 ** 2))
    n7 = 0.271219247759287*s**3 + 1.9214235718342*s**2 + 29.9262721326149*s + 131.844981042045
    # # do a manual partial fraction for d6 / n6 here...
    # print(d6)
    # a1, a2, a3, a4 = sympy.symbols('a1 a2 a3 a4')
    # print(sympy.expand(n7 * s * a1 + (a2 * s**2 + a3 * s**1 + a4) * (s**2 + w5**2)))
    # tmp_expr = sympy.collect(sympy.expand(n7 * s * a1 + (a2 * s**2 + a3 * s**1 + a4) * (s**2 + w5**2) - d6), s)
    # print(tmp_expr)
    # print(tmp_expr.coeff(s, 4))
    # solve AA * xx = bb
    bb = [245.933760498865, 1382.10350312125, 59257.3118804709, 228654.6870933, 2165232.39530017]
    AA = [[0.271219247759287, 1, 0, 0],
          [1.9214235718342, 0, 1, 0],
          [29.9262721326149, 294.61890839799, 0, 1],
          [131.844981042045, 0, 294.61890839799, 0],
          [0, 0, 0, 294.61890839799]]
    xx, *other = np.linalg.lstsq(AA, bb)
    # a1 = xx[0] for the shunt LC circuit
    C6 = 1/xx[0]
    L6 = xx[0]/(w5**2)
    d7 = xx[1] * s ** 2 + xx[2] * s + xx[3]
    # print(d7)
    z7 = d7 / n7
    # shunt C7 for (0.271219247759287/134.423988547205) e-9 F
    C7 = (0.271219247759287/134.423988547205)
    d8 = sympy.simplify(n7 - C7 * s * d7)
    d8 = 0.726760181741815*s**2 + 15.0980996256113*s + 131.844981042045
    # print(d8)
    z8 = d7 / d8

    # 4. repeat again
    all_freq = np.linspace(0, 40, 10000)
    all_freq = np.linspace(9.21, 9.215, 10000)
    z8_f = sympy.lambdify(s, z8, "numpy")
    z8_num = z8_f(1j * all_freq)
    idx = np.argmin(z8_num.real)
    w8 = all_freq[idx]
    r8 = z8_num.real[idx]
    x8 = z8_num.imag[idx]
    plt.plot(all_freq, z8_num.real)
    plt.grid(True)
    # minimum between 9.21 and 9.215
    # print(w8, r8, x8)
    z9 = z8 - r8
    # x8 > 0, case B
    y9 = sympy.simplify(1/z9)
    y9_f = sympy.lambdify(s, y9, "numpy")
    y9_num = y9_f(1j*w8).imag
    # print(y9_num, y9_num/w8)
    # take out shunt negative capacitance C9 y9_num / w8 = -0.0027802422089461027 e-9
    # also take out the zero at w8 = 9.213003300330033
    C9 = y9_num/w8
    n9, d9 = sympy.fraction(sympy.simplify(y9 - s * C9))
    # print(n6)
    # print(d6)
    n10 = sympy.apart(n9 / (s ** 2 + w8 ** 2))
    n10 = 0.33427684995401*s + 1.5533207672841
    # # do a manual partial fraction for d9 / n9 here...
    # print(d9)
    # a1, a2 = sympy.symbols('a1 a2')
    # print(sympy.expand(n10 * s * a1 + a2 * (s**2 + w8**2)))
    # tmp_expr = sympy.collect(sympy.expand(n10 * s * a1 + a2 * (s**2 + w8**2) - d9), s)
    # print(tmp_expr)
    # print(tmp_expr.coeff(s, 2))
    # solve AA * xx = bb
    bb = [120.232995844173, 297.298049386714, 4774.81017843864]
    AA = [[0.33427684995401, 1],
          [1.5533207672841, 0],
          [0, 84.8794298118921]]
    xx, *other = np.linalg.lstsq(AA, bb)
    # a1 = xx[0] for the shunt LC circuit
    C9 = 1/xx[0]
    L9 = xx[0]/(w8**2)
    d10 = xx[1]
    # print(d10)
    z10 = d10 / n10
    # shunt C10 for (0.33427684995401/56.2540322078259) e-9 F
    C10 = (0.33427684995401/56.2540322078259)
    d11 = sympy.simplify(n10 - C10 * s * d10)
    d11 = 1.5533207672841
    # print(d11)
    z11 = d10 / d11
    # z11 = 36.215335166208526
    # done

    # summarize:
    CC0 = (1/814.7909188) * 1e-9
    RR1 = r1
    CC1 = y2_num/w1 * 1e-9
    CC2 = C3 * 1e-9
    LL2 = L3 * 1e-9
    CC3 = (0.380730360161106/247.97122460336) * 1e-9
    RR4 = r5
    CC4 = y6_num/w5 * 1e-9
    CC5 = C6 * 1e-9
    LL5 = L6 * 1e-9
    CC6 = (0.271219247759287/134.423988547205) * 1e-9
    RR7 = r8
    CC7 = y9_num/w8 * 1e-9
    CC8 = C9 * 1e-9
    LL8 = L9 * 1e-9
    CC9 = (0.33427684995401/56.2540322078259) * 1e-9
    RR = 36.215335166208526

    # transform to coupled inductance form
    LLL1 = CC3 / (CC1 + CC3) * LL2
    LLL2 = CC2 / (CC1 + CC3) * LL2
    LLL3 = CC1 / (CC1 + CC3) * LL2
    CCC2 = (CC1 + CC3)
    MMM2 = LLL2
    LLL1 = LLL1 + LLL2
    LLL3 = LLL3 + LLL2

    LLL4 = CC6 / (CC4 + CC6) * LL5
    LLL5 = CC5 / (CC4 + CC6) * LL5
    LLL6 = CC4 / (CC4 + CC6) * LL5
    CCC5 = (CC4 + CC6)
    MMM5 = LLL5
    LLL4 = LLL4 + LLL5
    LLL6 = LLL6 + LLL5

    LLL7 = CC9 / (CC7 + CC9) * LL8
    LLL8 = CC8 / (CC7 + CC9) * LL8
    LLL9 = CC7 / (CC7 + CC9) * LL8
    CCC8 = (CC7 + CC9)
    MMM8 = LLL8
    LLL7 = LLL7 + LLL8
    LLL9 = LLL9 + LLL8

    return


def dipole():
    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi

    # Try to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=3, n_iter=20, s_inf=-1)
    # f_out = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=-1, bound_wt=0.3)
    # f_out = vecfit.bound_tightening(s_data, cs)  # np.inf, -1
    f_out, log = vecfit.bound_tightening_sweep(s_data, cs)

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
    # f_out = vecfit.bound_tightening(s_data, cs)
    f_out, log = vecfit.bound_tightening_sweep(s_data, cs)

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


def long_dipole_paper():
    s1p_file = './resource/single_dipole_wideband.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi
    z0 = 50
    z_data = (1 + s_data) / (1 - s_data) * z0
    fz = interp1d(freq, z_data)
    fr = interp1d(freq, np.real(z_data))
    fx = interp1d(freq, np.imag(z_data))
    f0 = 2.4e9
    f1 = optimize.bisect(fx, 2e9, 2.5e9)
    f2 = optimize.bisect(fx, 4e9, 4.2e9)
    f3 = optimize.bisect(fx, 6.8e9, 7.2e9)
    f4 = optimize.bisect(fx, 8.5e9, 8.8e9)

    # Equivalent circuit of dipole antenna of arbitrary length
    # proposed structure: C0 + L0 + (R1 // L1 // C1) + (R2 // L2 // C2)
    C0 = -1 / (2 * np.pi * f0/10 * fx(f0/10))  # determined by 0.1 * lambda
    L0 = 1 / (2 * np.pi * (3*f1)) ** 2 / C0  # determined by 0.5 * lambda and C0
    R1 = fr(f2)  # determined by 1 * lambda
    C1 = 1e-13  # determined by 0.5 * lambda and 1 * lambda
    L1 = 1 / (2 * np.pi * f2) ** 2 / C1  # determined by 0.5 * lambda and 1 * lambda
    R2 = fr(f4)  # determined by 2 * lambda
    C2 = 1.2e-13  # determined by 1.5 * lambda and 2 * lambda
    L2 = 1 / (2 * np.pi * (1.018*f4)) ** 2 / C2  # determined by 1.5 * lambda and 2 * lambda

    p1 = (-1/R1/C1 + cmath.sqrt(1/R1**2/C1**2 - 4/L1/C1)) / 2
    p2 = (-1/R1/C1 - cmath.sqrt(1/R1**2/C1**2 - 4/L1/C1)) / 2
    r1 = 1/C1 / (1 - p2/p1)
    r2 = 1/C1 / (1 - p1/p2)
    p3 = (-1/R2/C2 + cmath.sqrt(1/R2**2/C2**2 - 4/L2/C2)) / 2
    p4 = (-1/R2/C2 - cmath.sqrt(1/R2**2/C2**2 - 4/L2/C2)) / 2
    r3 = 1/C2 / (1 - p4/p3)
    r4 = 1/C2 / (1 - p3/p4)
    pole_model = [0, p1, p2, p3, p4]
    zero_model = [1 / C0, r1, r2, r3, r4]
    z_model = vecfit.RationalFct(pole_model, zero_model, 0, L0)
    s_tmp = (z_model.model(cs) - z0) / (z_model.model(cs) + z0)
    s_model = vecfit.fit_s(s_tmp, cs, n_pole=6, n_iter=20, s_inf=1)
    paper_bound, paper_bw = s_model.bound(np.inf, f0=2.4e9)
    paper_bound_error = s_model.bound_error(s_data, cs, reflect=np.inf)

    zl = z_model.model(cs)
    zl2 = 1/(cs*C0) + cs*L0 + 1/(1/R1 + 1/(cs*L1) + cs*C1) + 1/(1/R2 + 1/(cs*L2) + cs*C2)
    # zl2 = 1/(cs*C0) + cs*L0 + 1/(1/R1 + 1/(cs*L1) + cs*C1)
    # zl2 = 1/(cs*C0) + cs*L0
    # zl2 = 1/(1/R1 + 1/(cs*L1) + cs*C1)

    # Use algorithm to fit S
    # f_out = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=1)
    # f_out = vecfit.bound_tightening(s_data, cs)
    f_out, log = vecfit.bound_tightening_sweep(s_data, cs, np.inf)  # -1
    z_fit = (1 + f_out.model(cs)) / (1 - f_out.model(cs)) * z0
    bound, bw = f_out.bound(np.inf, f0=2.4e9)
    bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)

    # # try to draw a model for the dipole model
    # z_fit_model = vecfit.fit_z(z_fit, cs, n_pole=7, n_iter=20, has_const=False, has_linear=False)
    # axz = vecfit.plot_freq_resp(cs, z_fit, y_scale='db')
    # z_fit_model.plot(cs, ax=axz, y_scale='db', linestyle='--')
    # vecfit.plot_freq_resp(cs, z_fit-z_fit_model.model(cs), y_scale='db')
    # plt.show()

    # Use manual algorithm to fit S
    # f_out2 = vecfit.fit_s(s_data, cs, n_pole=6, n_iter=20, s_inf=1)
    f_out2 = vecfit.fit_s_v2(s_data, cs, n_pole=6, n_iter=20, s_inf=1)
    z_fit2 = (1 + f_out2.model(cs)) / (1 - f_out2.model(cs)) * z0
    bound2, bw2 = f_out2.bound(np.inf, f0=2.4e9)
    bound_error2 = f_out2.bound_error(s_data, cs, reflect=np.inf)

    # plots
    ax = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    vecfit.plot_freq_resp(cs, s_tmp, ax=ax, y_scale='db', linestyle='--')
    f_out.plot(cs, ax=ax, y_scale='db', linestyle='--')
    # f_out2.plot(cs, ax=ax, y_scale='db', linestyle='--')
    ax.set_ylabel(r'Reflection coefficient $\Gamma$ (dB)')
    ax.legend(['Simulation', 'Paper [18?]', 'Algorithm 2'])
    # ax.legend(['Simulation', 'Paper [18?]', 'Algorithm 2', 'Impedance Method'])
    ax.grid(True, which='both', linestyle='--')

    # calculate integrated error
    err_tmp = vecfit.num_integral(cs.imag, np.abs(s_tmp - s_data))
    err_f_out = vecfit.num_integral(cs.imag, np.abs(f_out.model(cs) - s_data))
    err_f_out2 = vecfit.num_integral(cs.imag, np.abs(f_out2.model(cs) - s_data))

    # bound values
    print('Paper:')
    print('Poles {}, bound {:.5e}, error {:.5e}, sum {:.5e}, integrated error {:.5e}'.format(len(s_model.pole), paper_bound, paper_bound_error, paper_bound + paper_bound_error, err_tmp))
    print('Algorithm:')
    print('Poles {}, bound {:.5e}, error {:.5e}, sum {:.5e}, integrated error {:.5e}'.format(len(f_out.pole), bound, bound_error, bound + bound_error, err_f_out))
    print('Manual comparison:')
    print('Poles {}, bound {:.5e}, error {:.5e}, sum {:.5e}, integrated error {:.5e}'.format(len(f_out2.pole), bound2, bound_error2, bound2 + bound_error2, err_f_out2))


def dipole_bound_vs_pole():
    s1p_file = './resource/single_dipole.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi
    s_all = np.logspace(np.log10(np.min(np.abs(cs))) - 3, np.log10(np.max(np.abs(cs))) + 3, int(1e5)) * 1j

    # plot 1: bound vs pole with reflection inf
    # ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 11):
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=i, n_iter=20, s_inf=-1)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        print('Pole {}, passive {}, stable {}'.format(i, passive, f_out.stable))
        bound, bw = f_out.bound(np.inf, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        # f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
        # vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(pole_list, bound_list, '--')
    ax.plot(pole_list, bound_error_list, '--')
    ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Number of poles $N_p$')
    ax.set_ylabel(r'Bound ($s_0=\infty$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')

    # plot 2: bound vs pole with reflection 0
    # ax2 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 13):
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=i, n_iter=20, s_dc=1)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        print('Pole {}, passive {}, stable {}'.format(i, passive, f_out.stable))
        bound, bw = f_out.bound(0, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=0)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        # f_out.plot(cs, ax=ax2, y_scale='db', linestyle='--')
        # vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax2, y_scale='db', linestyle='--')

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(pole_list, bound_list, '--')
    ax.plot(pole_list, bound_error_list, '--')
    ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Number of poles $N_p$')
    ax.set_ylabel(r'Bound ($s_0=0$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')

    # plot 3: bound vs weight with reflection inf
    wt_list = []
    bound_list = []
    bound_error_list = []
    wt = 0
    wt_step = 1
    while wt <= 100:
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=3, n_iter=20, s_inf=-1, bound_wt=wt)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        if passive and f_out.stable:
            bound, bw = f_out.bound(np.inf, f0=2.4e9)
            bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
            wt_list.append(wt)
            bound_list.append(bound)
            bound_error_list.append(bound_error)
            wt += wt_step
        else:
            print('Stop at weight {:g}, passive {}, stable {}'.format(wt, passive, f_out.stable))
            break

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(wt_list, bound_list, '--')
    ax.plot(wt_list, bound_error_list, '--')
    ax.plot(wt_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Weight $\alpha$')
    ax.set_ylabel(r'Bound ($s_0=\infty$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')

    # plot 4: bound vs weight with reflection 0
    wt_list = []
    bound_list = []
    bound_error_list = []
    wt = 0
    wt_step = 0.001
    while True:
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=5, n_iter=20, s_dc=1, bound_wt=wt)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        if passive and f_out.stable:
            bound, bw = f_out.bound(0, f0=2.4e9)
            bound_error = f_out.bound_error(s_data, cs, reflect=0)
            wt_list.append(wt)
            bound_list.append(bound)
            bound_error_list.append(bound_error)
            wt += wt_step
        else:
            print('Stop at weight {:g}, passive {}, stable {}'.format(wt, passive, f_out.stable))
            break

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(wt_list, bound_list, '--')
    ax.plot(wt_list, bound_error_list, '--')
    ax.plot(wt_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Weight $\alpha$')
    ax.set_ylabel(r'Bound ($s_0=0$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')


def long_dipole_bound_vs_pole():
    s1p_file = './resource/single_dipole_wideband.s1p'
    freq, n, z_data, s_data, z0_data = vecfit.read_snp(s1p_file)
    cs = freq*2j*np.pi
    s_all = np.logspace(np.log10(np.min(np.abs(cs))) - 3, np.log10(np.max(np.abs(cs))) + 3, int(1e5)) * 1j

    # plot 1: bound vs pole with reflection inf
    # ax1 = vecfit.plot_freq_resp(cs, s_data, y_scale='db')
    pole_list = []
    bound_list = []
    bound_error_list = []
    for i in range(1, 15):
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=i, n_iter=20, s_inf=-1)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        print('Pole {}, passive {}, stable {}'.format(i, passive, f_out.stable))
        bound, bw = f_out.bound(np.inf, f0=2.4e9)
        bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
        pole_list.append(i)
        bound_list.append(bound)
        bound_error_list.append(bound_error)
        # f_out.plot(cs, ax=ax1, y_scale='db', linestyle='--')
        # vecfit.plot_freq_resp(cs, s_data - f_out.model(cs), ax=ax1, y_scale='db', linestyle='--')

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(pole_list, bound_list, '--')
    ax.plot(pole_list, bound_error_list, '--')
    ax.plot(pole_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Number of poles $N_p$')
    ax.set_ylabel(r'Bound ($s_0=\infty$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')

    # plot 2: bound vs weight with reflection inf
    wt_list = []
    bound_list = []
    bound_error_list = []
    wt = 0
    wt_step = 0.2
    while wt <= 20:
        f_out = vecfit.fit_s_v2(s_data, cs, n_pole=7, n_iter=20, s_inf=-1, bound_wt=wt)
        passive = np.all(np.abs(f_out.model(s_all)) <= 1)
        if passive and f_out.stable:
            bound, bw = f_out.bound(np.inf, f0=2.4e9)
            bound_error = f_out.bound_error(s_data, cs, reflect=np.inf)
            wt_list.append(wt)
            bound_list.append(bound)
            bound_error_list.append(bound_error)
            wt += wt_step
        else:
            print('Stop at weight {:g}, passive {}, stable {}'.format(wt, passive, f_out.stable))
            break

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(wt_list, bound_list, '--')
    ax.plot(wt_list, bound_error_list, '--')
    ax.plot(wt_list, np.array(bound_list) + np.array(bound_error_list))
    ax.set_xlabel(r'Weight $\alpha$')
    ax.set_ylabel(r'Bound ($s_0=\infty$)')
    ax.legend([r'$B$', r'$\delta B$', r'$B+\delta B$'])
    ax.grid(True, which='both', linestyle='--')


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
    # f_even = vecfit.bound_tightening(s_even, cs)
    # f_odd = vecfit.bound_tightening(s_odd, cs)
    f_even, log_even = vecfit.bound_tightening_sweep(s_even, cs, np.inf)
    f_odd, log_odd = vecfit.bound_tightening_sweep(s_odd, cs, np.inf)

    bound_even, bw_even = f_even.bound(np.inf, f0=2.4e9)
    bound_error_even = f_even.bound_error(s_even, cs, reflect=np.inf)
    bound_odd, bw_odd = f_odd.bound(np.inf, f0=2.4e9)
    bound_error_odd = f_odd.bound_error(s_odd, cs, reflect=np.inf)
    print('Manual even/odd mode decomposition:')
    print('Even mode, # poles {}, bound {:.5e}, bound error {:.5e}, sum {:.5e}'.format(len(f_even.pole), bound_even, bound_error_even, bound_even + bound_error_even))
    print('Odd mode, # poles {}, bound {:.5e}, bound error {:.5e}, sum {:.5e}'.format(len(f_odd.pole), bound_odd, bound_error_odd, bound_odd + bound_error_odd))
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


def circuit_four():
    # build the circuit
    L0 = 0.1e-9
    R0 = 1
    # R11 = 15.85
    R11 = 45.85
    L11 = 11.52e-9
    C11 = 2.25e-12
    L12 = 1.49e-9
    C12 = 2.3e-12
    L13 = 1.77e-9
    C13 = 0.5e-12
    L24 = 8.34e-9
    C24 = 5.65e-12
    L34 = 0.5e-9
    C34 = 4.42e-12

    # build Y matrix
    freq = np.linspace(1e9, 5e9, 1000)
    cs = freq*2j*np.pi
    Y11 = 1/R11 + C11*cs + 1/(L11*cs)
    Y12 = C12*cs + 1/(L12*cs)
    Y13 = C12*cs + 1/(L12*cs)
    Y24 = C12*cs + 1/(L12*cs)
    Y34 = C12*cs + 1/(L12*cs)
    Y_mx = np.zeros([4, 4, len(freq)], dtype=np.complex128)
    Y_mx[0, 0, :] = Y11 + Y12 + Y13
    Y_mx[0, 1, :] = -Y12
    Y_mx[1, 0, :] = -Y12
    Y_mx[0, 2, :] = -Y13
    Y_mx[2, 0, :] = -Y13
    Y_mx[1, 1, :] = Y11 + Y12 + Y24
    Y_mx[1, 3, :] = -Y24
    Y_mx[3, 1, :] = -Y24
    Y_mx[2, 2, :] = Y11 + Y13 + Y34
    Y_mx[2, 3, :] = -Y34
    Y_mx[3, 2, :] = -Y34
    Y_mx[3, 3, :] = Y11 + Y24 + Y34
    Z_mx = np.zeros([4, 4, len(freq)], dtype=np.complex128)
    Z0 = R0 + L0*cs
    for i in range(len(freq)):
        Z_mx[:, :, i] = np.linalg.inv(Y_mx[:, :, i])
    Z_mx[0, 0, :] += Z0
    Z_mx[1, 1, :] += Z0
    Z_mx[2, 2, :] += Z0
    Z_mx[3, 3, :] += Z0
    S_mx = np.zeros([4, 4, len(freq)], dtype=np.complex128)
    z0 = 50
    n = 4
    for i in range(len(freq)):
        S_mx[:, :, i] = np.matrix(Z_mx[:, :, i] / z0 - np.identity(n)) * np.linalg.inv(np.matrix(Z_mx[:, :, i] / z0 + np.identity(n)))
    
    # feed S matrix into algorithm
    s_matrix, bound = vecfit.mode_fitting(S_mx, cs, True)
    bound_error = s_matrix.bound_error(S_mx, cs, reflect=np.inf)
    ant_integral = vecfit.bound_integral(S_mx, cs, np.inf)
    print('Bound is {:.5e}, Bound error is {:.5e}, The integral of the antenna is {:.5e}'.format(bound, bound_error, ant_integral))

    fig = plt.figure(figsize=(8, 5.5))
    ax1 = fig.add_subplot(111)
    for i in range(4):
        vecfit.plot_freq_resp(cs, S_mx[0, i, :], ax=ax1, y_scale='db', color=colors(i))
    ax1.legend(['$S_{11}$', '$S_{12}$', '$S_{13}$', '$S_{14}$'])
    for i in range(4):
        vecfit.plot_freq_resp(cs, s_matrix.model(cs)[0, i, :], ax=ax1, y_scale='db', linestyle='--', color=colors(i))


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
    # axs = vecfit.plot_freq_resp_matrix(cs, s_data, y_scale='db')
    # s_matrix.plot(cs, axs=axs, y_scale='db', linestyle='--')
    fig = plt.figure(figsize=(8, 5.5))
    # ax1 = fig.add_subplot(211)
    # for i in range(4):
    #     vecfit.plot_freq_resp(cs, s_data[i, i, :], ax=ax1, y_scale='db', color=colors(i))
    #     vecfit.plot_freq_resp(cs, s_matrix.model(cs)[i, i, :], ax=ax1, y_scale='db', linestyle='--', color=colors(i))
    # ax1 = fig.add_subplot(212)
    # idx = 0
    # for i in range(4):
    #     for j in range(1, i):
    #         vecfit.plot_freq_resp(cs, s_data[i, j, :], ax=ax1, y_scale='db', color=colors(idx))
    #         vecfit.plot_freq_resp(cs, s_matrix.model(cs)[i, j, :], ax=ax1, y_scale='db', linestyle='--', color=colors(idx))
    #         idx += 1
    
    ax1 = fig.add_subplot(111)
    for i in range(4):
        vecfit.plot_freq_resp(cs, s_data[0, i, :], ax=ax1, y_scale='db', color=colors(i))
    ax1.legend(['$S_{11}$', '$S_{12}$', '$S_{13}$', '$S_{14}$'])
    for i in range(4):
        vecfit.plot_freq_resp(cs, s_matrix.model(cs)[0, i, :], ax=ax1, y_scale='db', linestyle='--', color=colors(i))


if __name__ == '__main__':
    # example1()
    # example2()
    # single_siw()
    # coupled_siw_even_odd()
    # coupled_siw()
    # transmission_line_model()
    # transmission_line_model_vs_freq_range()
    # dipole()
    # dipole_paper()
    # dipole_bound_vs_pole()
    # long_dipole_paper()
    # long_dipole_bound_vs_pole()
    # coupled_dipole()
    # skycross_antennas()
    circuit_four()
    # two_ifa()
    # four_ifa()

    plt.show()
    # plot_save()


