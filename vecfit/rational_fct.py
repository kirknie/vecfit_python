# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:31:51 2018

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import skrf as rf
from . import vector_fitting


class RationalFct:
    def __init__(self, pole, residue, const=None, linear=None):
        self.pole = np.array(pole)
        self.residue = np.array(residue)
        self.const = const
        self.linear = linear

    def __add__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, other.residue])
        d = self.const
        if other.const is not None:
            if d is None:
                d = other.const
            else:
                d += other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = other.linear
            else:
                h += other.linear
        return RationalFct(p, r, d, h)

    def __sub__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, -other.residue])
        d = self.const
        if other.const is not None:
            if d is None:
                d = -other.const
            else:
                d -= other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = -other.linear
            else:
                h -= other.linear
        return RationalFct(p, r, d, h)

    def model(self, s):
        """
        Return the function value at s
        :param s: input argument, 1D array
        :return: function value at s, 1D array
        """
        s = np.array(s)
        f = sum(r/(s-p) for (p, r) in zip(self.pole, self.residue))
        if self.const is not None:
            f += self.const
        if self.linear is not None:
            f += s*self.linear
        return f

    def plot(self, s, ax=None, x_scale=None, y_scale=None, **kwargs):
        x = np.abs(s)/2/np.pi
        y = self.model(s)
        fig = plt if ax is None else ax

        plt_fct = fig.plot
        if x_scale == 'linear':
            plt_fct = fig.plot
        elif x_scale == 'log':
            plt_fct = fig.semilogx
        if y_scale == 'linear':
            pass
        elif y_scale == 'db':
            y = 20 * np.log10(np.abs(y))

        plt_fct(x, y, **kwargs)
        fig.grid(True, which='both', linestyle='--')
        fig.set_xlabel('Frequency (Hz)')
        fig.set_ylabel('Amplitude (dB)')
        return fig, ax

    def zero(self):
        """
        Calculate zeros of the function
        :return: zeros, 1D array
        """
        if self.const:
            return vector_fitting.calculate_zero(self.pole, self.residue, self.const)
        else:
            raise RuntimeError('Do not have a constant term')

    def bound(self, reflect, tau=0.3162278, f0=0):
        """
        Calculate the bound and maximum bandwidth of the S-parameter model
        :param reflect: reflection point
        :param tau: threshold for pass band, for bw calculation
        :param f0: center frequency of pass band, for bw calculation
        :return: the bound and maximum bandwidth
        """
        bw = np.nan
        if reflect == 0:
            bound = -np.pi / 2 * (sum(1 / self.pole) + sum(1 / self.zero()))
            bound = bound.real
            if f0 != 0:
                bw = bound / 2 / np.pi / np.log(1 / tau) * (2 * np.pi * f0) ** 2
        elif reflect == np.inf:
            bound = -np.pi / 2 * (sum(self.pole) + sum(self.zero()))
            bound = bound.real
            bw = bound / 2 / np.pi / np.log(1 / tau)
        else:
            raise RuntimeError('Bound calculation unsupported yet!')
        return bound, bw

    def bound_error(self, f, s, reflect, tau=0.3162278):
        """
        Calculate the bound error of the S-parameter model
        :param f: the true S-parameter of load at s, 1D array
        :param s: the input complex frequency
        :param reflect: reflection point
        :param tau: threshold for pass band
        :return: the bound error
        """
        f_fit = self.model(s)
        f_error = f_fit - f
        # Calculate rho first (single load)
        rho = (2 * np.abs(f_error)) / (1 - np.power(np.abs(f), 2))
        int_fct = f_integral(s.imag, reflect) / 2 * np.log(1 + (1 - tau ** 2) / tau ** 2 * rho)
        # delta_b = np.sum((int_fct[:-1] + int_fct[1:]) / 2 * (s.imag[1:] - s.imag[:-1]))
        delta_b = num_integral(s.imag, int_fct)
        return delta_b

    def bound_integral(self, s, reflect):
        """
        Calculate the bound integral
        :param s: complex frequency, 1D array
        :param reflect: reflection point
        :return: the integral value of the load, to be compared with the bound
        """
        return bound_integral(self.model(s), s, reflect)

    def plot_improved_bound(self, real_limit, imag_limit, ax=None):
        fig = plt if ax is None else ax

        sample = 2000
        x = np.linspace(-real_limit, 0, sample)
        y = np.linspace(-imag_limit, imag_limit, 2 * sample)
        xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
        s = xv + 1j * yv
        f = self.model(s)
        zero = self.zero()

        # do the plot
        fig.contourf(x, y, 2 - np.abs(f), [1, 2])
        fig.plot(self.pole.real, self.pole.imag, 'x')
        fig.plot(zero.real, zero.imag, 'o')
        fig.axis([-real_limit, 0, -imag_limit, imag_limit])
        fig.set_xlabel('Re{s}')
        fig.set_ylabel('Im{s}')

        return s, f


def f_integral(w, reflect):
    """
    Generate the function inside of the bound integral
    :param w: input radian frequency, 1D array
    :param reflect: reflection point
    :return: numerical value of the function at w
    """
    if reflect == np.inf:
        f = np.ones(len(w))
    elif reflect.real > 0.0:
        f = 1/2 * (1/(reflect - 1j*w) + 1/(reflect + 1j*w)).real
    elif reflect.real == 0.0:
        f = 1/2 * (np.power(reflect.imag - w, -2) + np.power(reflect.imag - w, -2))
    else:
        raise RuntimeError('Invalid reflection point!')
    return f


def num_integral(x, y):
    """
    Calculate the numerical integral value
    :param x: x, 1D array
    :param y: y, 1D array
    :return: integral value
    """
    return np.sum((y[:-1] + y[1:]) / 2 * (x[1:] - x[:-1]))


def bound_integral(gamma, s, reflect):
    """
    Calculate the bound integral
    :param gamma: power loss ratio, 1D array
    :param s: complex frequency, 1D array
    :param reflect: reflection point
    :return: the integral value of the load, to be compared with the bound
    """
    return num_integral(s.imag, f_integral(s.imag, reflect) * np.log(1/np.abs(gamma)))


def get_z0(snp_file):
    """
    Get characteristic impedance Z0 from .snp file
    :param snp_file: .snp file name, str
    :return: Z0, nxl array
    """
    n = int(snp_file[-2])
    if '.s' + str(n) + 'p' not in snp_file:
        raise RuntimeError('Error: input file is not snp touchstone file')
    z0 = [[] for i in range(n)]
    with open(snp_file) as f:
        for l in f:
            if '! Port Impedance' in l:
                z0_str = list(filter(None, l[len('! Port Impedance'):].split(' ')))
                for i in range(n):
                    z0[i].append(float(z0_str[2*i]) + 1j*float(z0_str[2*i+1]))
    z0 = np.array(z0)
    if n == 1:
        z0 = z0.reshape(len(z0[0]))
    return np.array(z0)


def s2z(s, z0):
    """
    Convert S-matrix to Z-matrix
    :param s: S-matrix, 1D array for n=1, nxnxl for nxn matrix
    :param z0: characteristic impedance, double or 1D array for n=1, nxl array for nxn matrix
    :return: Z-matrix, 1D array for n=1, nxnxl for nxn matrix
    """
    if s.ndim == 1:  # one port
        z = z0 * (1+s) / (1-s)
    elif s.ndim == 3:
        n = s.shape[0]
        l = s.shape[2]
        z = np.zeros(s.shape, dtype=s.dtype)
        for i in range(l):
            z0_sq = np.matrix(np.diag(np.sqrt(z0[:, i])))
            s_ele = s[:, :, i]
            z[:, :, i] = z0_sq * np.matrix(numpy.identity(n)+s_ele) * np.linalg.inv(np.matrix(numpy.identity(n)-s_ele)) * z0_sq
    else:
        raise RuntimeError('Error: unexpected S-parameter format')
    return z


def read_snp(snp_file):
    """
    Read the .snp file for frequency, n, Z-matrix, S-matrix, and characteristic impedance
    :param snp_file: .snp file name, str
    :return: all parameters from .snp file
    """
    ant_data = rf.Network(snp_file)
    z0 = get_z0(snp_file)
    n = ant_data.number_of_ports
    freq = ant_data.f
    l = len(freq)
    if n == 1:
        s = ant_data.s.reshape(l)
    else:
        s = np.zeros([n, n, l], dtype=np.complex128)
        for i in range(l):
            s[:, :, i] = ant_data.s[i, :, :]
    z = s2z(s, z0)
    return freq, n, z, s, z0


class RationalMtx:
    def __init__(self, pole, residue, const=None, linear=None):
        if residue.ndim == 3 and residue.shape[0] == residue.shape[1] and residue.shape[2] == len(pole):
            self.ndim = residue.shape[0]
            self.pole = np.array(pole)
            self.residue = np.array(residue)
            self.const = const
            self.linear = linear
        else:
            raise RuntimeError('RationalMtx: Input format not expected')

    def __add__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, other.residue], axis=2)
        d = self.const
        if other.const is not None:
            if d is None:
                d = other.const
            else:
                d += other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = other.linear
            else:
                h += other.linear
        return RationalFct(p, r, d, h)

    def __sub__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, -other.residue], axis=2)
        d = self.const
        if other.const is not None:
            if d is None:
                d = -other.const
            else:
                d -= other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = -other.linear
            else:
                h -= other.linear
        return RationalFct(p, r, d, h)

    def model(self, s):
        s = np.array(s)
        ns = len(s)
        f = np.zeros([self.ndim, self.ndim, ns], dtype=np.complex128)
        for i, si in enumerate(s):
            f[:, :, i] = np.sum(self.residue[:, :, j] / (si - p) for j, p in enumerate(self.pole))
            if self.const is not None:
                f[:, :, i] += self.const
            if self.linear is not None:
                f[:, :, i] += si * self.linear
        return f

    def is_symmetric(self):
        if self.const is not None and not np.allclose(self.const, self.const.T):
            return False
        if self.linear is not None and not np.allclose(self.linear, self.linear.T):
            return False
        for i in range(np.size(self.residue, 2)):
            mtx = self.residue[:, :, i]
            if not np.allclose(mtx, mtx.T):
                return False
        return True

    def rank_one(self):
        if self.is_symmetric():
            pole_pair = vector_fitting.pair_pole(self.pole)
            new_residue = np.zeros([self.ndim, np.size(self.residue, 2)], dtype=self.residue.dtype)
            for i, pp in enumerate(pole_pair):
                s, u = takagi(self.residue[:, :, i])
                if pp == 0:
                    new_residue[:, i] = u[:, 0] * np.sqrt(s[0])
                elif pp == 1:
                    new_residue[:, i] = u[:, 0] * np.sqrt(s[0])
                    new_residue[:, i+1] = (u[:, 0] * np.sqrt(s[0])).conj()
            # new_const = None
            # new_linear = None
            # if self.const is not None:
            #     s, u = takagi(self.const)
            #     new_const = u[:, 0] * np.sqrt(s[0])
            # if self.linear is not None:
            #     s, u = takagi(self.linear)
            #     new_linear = u[:, 0] * np.sqrt(s[0])
            # return RationalRankOneMtx(self.pole, new_residue, new_const, new_linear)
            return RationalRankOneMtx(self.pole, new_residue, self.const, self.linear)
        else:
            raise RuntimeError('Asymmetric matrix is not supported yet!')


def mat2vec(f_mat, symmetric=True):
    """
    Convert a matrix into the vector form
    :param f_mat: nxnxl array
    :param symmetric: whether the matrix is symmetric
    :return: nx(n+1)/2-by-l array if symmetric, n^2-by-l if asymmetric
    """
    # Function to transform a symmetric matrix into a vector form
    # Input: [ndim, ndim, ns]
    # Output: [ndim*(ndim+1)/2, ns]
    dim_one = False
    if f_mat.ndim == 2:
        dim_one = True
        ns = 1
        f_mat = f_mat.reshape(list(f_mat.shape) + [1])
    elif f_mat.ndim == 3:
        ns = np.size(f_mat, 2)
    else:
        raise RuntimeError('Unexpected input data format!')
    ndim = np.size(f_mat, 0)

    idx = 0
    if symmetric:
        f_vec = np.zeros([ndim * (ndim + 1) // 2, ns], dtype=np.complex128)
        for i in range(ndim):
            for j in range(i, ndim):
                f_vec[idx, :] = f_mat[i, j, :]
                idx += 1
    else:
        f_vec = np.zeros([ndim * ndim, ns], dtype=np.complex128)
        for i in range(ndim):
            for j in range(ndim):
                f_vec[idx, :] = f_mat[i, j, :]
                idx += 1

    if dim_one:  # Output a single vector
        f_vec = f_vec.reshape(np.size(f_vec, 0))
    return f_vec


def vec2mat(f_vec, symmetric=True):
    """
    Convert a matrix from the vector form into the matrix form, inverse function of mat2vec
    :param f_vec: nx(n+1)/2-by-l array if symmetric, n^2-by-l if asymmetric
    :param symmetric: whether the matrix is symmetric
    :return: nxnxl array
    """
    # Function to transform a vector form into a symmetric matrix
    # Input: [ndim*(ndim+1)/2, ns]
    # Output: [ndim, ndim, ns]
    dim_one = False
    if f_vec.ndim == 1:
        dim_one = True
        ns = 1
        f_vec = f_vec.reshape(list(f_vec.shape) + [1])
    elif f_vec.ndim == 2:
        ns = np.size(f_vec, 1)
    else:
        raise RuntimeError('Unexpected input data format!')
    ndim = np.size(f_vec, 0)
    if symmetric:
        ndim = int(np.sqrt(ndim * 8 + 1) - 1) // 2  # calculate the dimension of the matrix
    else:
        ndim = int(np.sqrt(ndim))

    f_mat = np.zeros([ndim, ndim, ns], dtype=np.complex128)
    idx = 0
    if symmetric:
        for i in range(ndim):
            for j in range(i, ndim):
                f_mat[i, j, :] = f_vec[idx, :]
                if i != j:
                    f_mat[j, i, :] = f_vec[idx, :]
                idx += 1
    else:
        for i in range(ndim):
            for j in range(ndim):
                f_mat[i, j, :] = f_vec[idx, :]
                idx += 1

    if dim_one:
        f_mat = f_mat.reshape([ndim, ndim])
    return f_mat


class RationalRankOneMtx:
    def __init__(self, pole, residue, const=None, linear=None):
        if residue.ndim == 2 and residue.shape[1] == len(pole):
            self.ndim = residue.shape[0]
            self.pole = np.array(pole)
            self.residue = np.array(residue)  # residue is a list of vectors, not matrix
            self.const = const  # const is a matrix, not necessarily rank 1
            self.linear = linear  # linear is a matrix, not necessarily rank 1
        else:
            raise RuntimeError('RationalRankOneMtx: Input format not expected')

    def __add__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, other.residue], axis=1)
        d = self.const
        if other.const is not None:
            if d is None:
                d = other.const
            else:
                d += other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = other.linear
            else:
                h += other.linear
        return RationalFct(p, r, d, h)

    def __sub__(self, other):
        p = np.concatenate([self.pole, other.pole])
        r = np.concatenate([self.residue, -other.residue], axis=1)
        d = self.const
        if other.const is not None:
            if d is None:
                d = -other.const
            else:
                d -= other.const
        h = self.linear
        if other.linear is not None:
            if h is None:
                h = -other.linear
            else:
                h -= other.linear
        return RationalFct(p, r, d, h)

    def model(self, s):
        s = np.array(s)
        ns = len(s)
        f = np.zeros([self.ndim, self.ndim, ns], dtype=np.complex128)
        for i, si in enumerate(s):
            f[:, :, i] = np.sum(np.outer(self.residue[:, j], self.residue[:, j]) / (si - p) for j, p in enumerate(self.pole))
            if self.const is not None:
                # f[:, :, i] += np.outer(self.const, self.const)
                f[:, :, i] += self.const
            if self.linear is not None:
                # f[:, :, i] += si * np.outer(self.linear, self.linear)
                f[:, :, i] += si * self.linear
        return f

    def full_rank(self):
        pole_pair = vector_fitting.pair_pole(self.pole)
        new_residue = np.zeros([self.ndim, self.ndim, np.size(self.residue, 1)], dtype=self.residue.dtype)
        for i, pp in enumerate(pole_pair):
            new_residue[:, :, i] = np.outer(self.residue[:, i], self.residue[:, i])
        return RationalRankOneMtx(self.pole, new_residue, self.const, self.linear)


def takagi(a):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('The input matrix is not a square matrix!')
    if not np.allclose(a, a.T):
        raise ValueError('The input matrix is not symmetric!')
    ndim = a.shape[0]
    u, s, v = np.linalg.svd(a)
    if not np.allclose(np.abs(u/v.T), np.ones([ndim, ndim])):
        print('Warning: The Takagi factorization result may not be accurate!')

    phase_diff = np.angle(u/v.T)[0, :]
    rotation_mtx = np.diag(np.exp(-phase_diff/2*1j))
    uu = u.dot(rotation_mtx)

    # Should have a = uu * diag(s) * uu.T
    if not np.allclose(a, uu.dot(np.diag(s)).dot(uu.T)):
        print('Warning: The Takagi factorization result is not accurate! ')
    return s, uu


