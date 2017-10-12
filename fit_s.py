# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:12:55 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import vector_fitting
import fit_z


def fit_s(f, s, n_poles=10, n_iters=10, s_dc=0, s_inf=0):
    # Call fit_z function and do some post processing
    if s_dc and s_inf:
        raise RuntimeError('Does not support dc and inf reflection simultaneously!')
    elif s_dc:
        if s_dc == 1:  # 1-s is bounded
            poles, residues, d, h = fit_z.fit_z(1-f, s, n_poles, n_iters, has_d=1, has_h=0, reflect_z=[0])
            d = 1-d
            residues = -residues
        elif s_dc == -1:  # 1+s is bounded
            poles, residues, d, h = fit_z.fit_z(1+f, s, n_poles, n_iters, has_d=1, has_h=0, reflect_z=[0])
            d -= 1
        else:  # not supported
            raise RuntimeError('Does not support s_dc not +/-1!')
    elif s_inf:
        if s_inf == 1:  # 1-s is bounded
            poles, residues, d, h = fit_z.fit_z(1-f, s, n_poles, n_iters, has_d=0, has_h=0)
            d = 1-d
            residues = -residues
        elif s_inf == -1:  # 1+s is bounded
            poles, residues, d, h = fit_z.fit_z(1+f, s, n_poles, n_iters, has_d=0, has_h=0)
            d -= 1
        else:  # not supported
            raise RuntimeError('Does not support s_inf not +/-1!')
    else:  # no specific reflection point
        poles, residues, d, h = vector_fitting.vector_fitting(f, s, n_poles, n_iters, has_d=1, has_h=0)
    
    return poles, residues, d, h


if __name__ == '__main__':
    """
    Example to fit s
    """
    