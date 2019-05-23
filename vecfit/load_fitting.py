# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:19:55 2017

@author: kirknie
"""

import numpy as np
import copy
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
        for s_inf in [-1, 1]:
            f_out, f_valid = fit_s_tight(f, s, n_pole, n_iter, s_inf)
            if f_valid:
                break
        if f_valid:
            break
        n_pole += 1
    else:
        raise RuntimeError('Fail to fit the S-parameter!')

    return f_out


def fit_s_tight(f, s, n_pole=10, n_iter=10, s_inf=1):
    init_wt = 0.01
    wt_iter = 20
    error_limit = 10 ** (-20 / 20)  # -20 dB error limit
    s_all = np.logspace(np.log10(np.min(np.abs(s))) - 3, np.log10(np.max(np.abs(s))) + 3, int(1e5)) * 1j

    # initial wt is 0
    wt = 0.0
    wt_a = 0.0
    wt_b = 0.0
    backtrace = False
    for i in range(wt_iter):
        if s_inf == 1:  # 1-s is bounded
            f_model = vector_fitting_rescale(1 - f, s, n_pole, n_iter, has_const=False, has_linear=False, bound_wt=wt)
            f_model.const = 1
            f_model.residue = -f_model.residue
        elif s_inf == -1:  # 1+s is bounded
            f_model = vector_fitting_rescale(1 + f, s, n_pole, n_iter, has_const=False, has_linear=False, bound_wt=wt)
            f_model.const = -1
        else:  # not supported
            raise RuntimeError('Does not support s_inf not +/-1!')

        # check if the result is passive

        small_error = np.linalg.norm(f_model.model(s) - f, np.inf) < error_limit
        passive = np.all(np.abs(f_model.model(s_all)) <= 1)
        ok = small_error and passive
        if i == 0 and not ok:  # if active with wt=0, stop
            f_out = copy.copy(f_model)
            f_valid = False
            break
        elif ok:  # if passive, increase the wt
            wt_a = wt
            f_out = copy.copy(f_model)
            f_valid = True
            if i == 0:
                wt = init_wt
            elif backtrace is False:
                wt *= 2
            else:
                wt = (wt_a + wt_b) / 2
        elif not ok:  # if active, decrease the wt
            wt_b = wt
            backtrace = True
            wt = (wt_a + wt_b) / 2
    return f_out, f_valid


