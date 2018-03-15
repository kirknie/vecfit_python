# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:31:51 2018

@author: kirknie
"""


class RationalFct:
    def __init__(self, pole, residue, const=None, linear=None):
        self.pole = pole
        self.residue = residue
        self.const = const
        self.linear = linear

    def model(self, s):
        f = sum(r/(s-p) for (p, r) in zip(self.pole, self.residue))
        if self.const:
            f += self.const
        if self.linear:
            f += s*self.linear
        return f


