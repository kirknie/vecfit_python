# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:19:55 2017

@author: kirknie
"""

import numpy as np
from .vector_fitting import vector_fitting_rescale


def fit_z(f, s, n_pole=10, n_iter=10, has_const=True, has_linear=True, fixed_pole=None, reflect_z=None):
    f_model = vector_fitting_rescale(f, s, n_pole, n_iter, has_const=has_const, has_linear=has_linear, fixed_pole=fixed_pole, reflect_z=reflect_z)
    return f_model


def fit_s(f, s, n_pole=10, n_iter=10, s_dc=None, s_inf=None, bound_wt=None):
    # Call vector_fitting function and do some post processing
    if s_dc and s_inf:
        raise RuntimeError('Does not support dc and inf reflection simultaneously!')
    elif s_dc:
        if s_dc == 1:  # 1-s is bounded
            f_model = vector_fitting_rescale(1-f, s, n_pole, n_iter, has_const=True, has_linear=False, reflect_z=0)
            f_model.const = 1 - f_model.const
            f_model.residue = -f_model.residue
        elif s_dc == -1:  # 1+s is bounded
            f_model = vector_fitting_rescale(1+f, s, n_pole, n_iter, has_const=True, has_linear=False, reflect_z=0)
            f_model.const -= 1
        else:  # not supported
            raise RuntimeError('Does not support s_dc not +/-1!')
    elif s_inf:
        if s_inf == 1:  # 1-s is bounded
            f_model = vector_fitting_rescale(1-f, s, n_pole, n_iter, has_const=False, has_linear=False, bound_wt=bound_wt)
            f_model.const = 1
            f_model.residue = -f_model.residue
        elif s_inf == -1:  # 1+s is bounded
            f_model = vector_fitting_rescale(1+f, s, n_pole, n_iter, has_const=False, has_linear=False, bound_wt=bound_wt)
            f_model.const = -1
        else:  # not supported
            raise RuntimeError('Does not support s_inf not +/-1!')
    else:  # no specific reflection point
        f_model = vector_fitting_rescale(f, s, n_pole, n_iter, has_const=True, has_linear=False)

    return f_model


def fit_s_auto(f, s):
    # automatically choose parameters for fit_s()
    n_iter = 20
    n_pole = 2
    max_pole = 50
    # loop until certain criterion is met: small error, passivity
    while n_pole <= max_pole:
        print('Modeling with {} poles...'.format(n_pole))
        for s_dc, s_inf in [(None, -1), (None, 1), (-1, None), (1, None)]:
            f_model = fit_s(f, s, n_pole, n_iter, s_dc, s_inf)  # do not put weight yet

            # check error
            error_limit = 10**(-20/20)
            small_error = np.linalg.norm(f_model.model(s) - f, np.inf) < error_limit
            s_all = np.logspace(np.log10(np.min(np.abs(s)))-3, np.log10(np.max(np.abs(s)))+3, int(1e5)) * 1j
            passive = np.all(np.abs(f_model.model(s_all)) <= 1)
            if small_error and passive:
                break
        if small_error and passive:
            break
        n_pole += 1
    else:
        raise RuntimeError('Fail to fit the S-parameter!')

    return f_model
