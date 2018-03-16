# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:31:51 2018

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt


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


