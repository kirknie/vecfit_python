# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:19:55 2017

@author: kirknie
"""

import numpy as np
import copy
from .vector_fitting import vector_fitting_rescale
from .matrix_fitting import joint_svd
from .rational_fct import RationalMtx


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


def bound_tightening(f, s):
    n_pole_max = 50
    error_limit = 10 ** (-20 / 20)  # -20 dB error limit
    norm_order = np.inf
    n_iter = 20  # Number of iterations for fitting
    wt_iter = 20  # Number of iterations for weight updating
    init_wt = 0.01  # Weight updating initial step

    s_all = np.logspace(np.log10(np.min(np.abs(s))) - 3, np.log10(np.max(np.abs(s))) + 3, int(1e5)) * 1j
    f_out = None

    for n_pole in range(1, n_pole_max+1):
        for s_inf in [-1, 1]:
            # initial wt is 0
            wt = 0.0
            wt_a = 0.0
            wt_b = 0.0
            start_tighten = False
            backtrace = False
            for i in range(wt_iter):
                f_model = fit_s(f, s, n_pole, n_iter, s_inf=s_inf, bound_wt=wt)

                # check if the result is passive
                small_error = np.linalg.norm(f_model.model(s) - f, norm_order) < error_limit
                passive = np.all(np.abs(f_model.model(s_all)) <= 1)
                valid = small_error and passive

                if valid and not start_tighten:
                    f_out = copy.copy(f_model)
                    start_tighten = True
                    wt = init_wt
                elif start_tighten:
                    if valid and not backtrace:
                        wt_a = wt
                        wt *= 2
                    elif valid:
                        wt_a = wt
                        wt = (wt_a + wt_b) / 2
                    else:
                        backtrace = True
                        wt_b = wt
                        wt = (wt_a + wt_b) / 2
                else:  # Increase number of poles
                    break
            else:
                return f_out
    return f_out


def mode_fitting(f, s, bound_output=False):
    wt = 1e3 * len(s)
    N = np.size(f, 0)
    A = np.concatenate([f, np.reshape(wt * np.identity(N), [N, N, 1])], 2)
    U, Sigma, Vh, dA, err_norm, orig_norm = joint_svd(A)
    all_model = []
    pole = np.zeros(0, dtype=np.complex128)
    residue = np.zeros([N, N, 0], dtype=np.complex128)
    const = np.zeros([N, N], dtype=np.complex128)

    for i in range(N):
        sigma_n = Sigma[i, i, :-1]
        sigma_model = bound_tightening(sigma_n, s)
        all_model.append(copy.copy(sigma_model))

        # re-build the matrix model
        pole = np.concatenate([pole, sigma_model.pole])
        ui = U[:, i]
        vhi = Vh[i, :]
        for r in sigma_model.residue:
            r_matrix = np.array(r * np.dot(ui, vhi))
            residue = np.concatenate([residue, np.reshape(r_matrix, [N, N, 1])], 2)
        const += sigma_model.const * np.dot(ui, vhi)
    # re-build the matrix model
    f_out = RationalMtx(pole, residue, const)
    if bound_output:
        total_bound = 0
        for i in range(N):
            bound, bw = all_model[i].bound(np.inf)
            total_bound += bound
            print('Mode {}, # poles {}, bound {:.5e}'.format(i, len(all_model[i].pole), bound))
            all_model[i].plot_improved_bound(max(-all_model[i].pole.real)*1.2, max(all_model[i].pole.imag)*1.2)
        total_bound /= N
        return f_out, total_bound
    else:
        return f_out


