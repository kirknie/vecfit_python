# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:31:51 2018

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
from .vector_fitting import calculate_zero


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
        s = np.array(s)
        f = sum(r/(s-p) for (p, r) in zip(self.pole, self.residue))
        if self.const is not None:
            f += self.const
        if self.linear is not None:
            f += s*self.linear
        return f

    def plot(self, s, ax=None, x_scale=None, y_scale=None, **kwargs):
        x = np.abs(s)
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
        plt.grid(True, linestyle='--')

    def zero(self):
        if self.const:
            return calculate_zero(self.pole, self.residue, self.const)
        else:
            raise RuntimeError('Do not have a constant term')

    def bound(self, reflect, tau=0.2, f0=0):
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

    def bound_error(self, f, s, reflect, tau=0.2):
        f_fit = self.model(s)
        f_error = f_fit - f
        # Calculate rho first (single load)
        rho = (2 * np.abs(f_error)) / (1 - np.power(np.abs(f), 2))
        int_fct = f_integral(s.imag, reflect) / 2 * np.log(1 + (1 - tau ** 2) / tau ** 2 * rho)
        # delta_b = np.sum((int_fct[:-1] + int_fct[1:]) / 2 * (s.imag[1:] - s.imag[:-1]))
        delta_b = num_integral(s.imag, int_fct)
        return delta_b

    def bound_integral(self, s, reflect):
        f = f_integral(s.imag, reflect) * np.log(1/self.model(s))
        return num_integral(s.imag, f)

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
        fig.xlabel('Re\{s\}')
        fig.ylabel('Im\{s\}')

        return s, f


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


def num_integral(x, y):
    return np.sum((y[:-1] + y[1:]) / 2 * (x[1:] - x[:-1]))




