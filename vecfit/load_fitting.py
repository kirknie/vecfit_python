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


def fit_s_v2(f, s, n_pole=10, n_iter=10, s_dc=None, s_inf=None, fit_wt=None, bound_wt=None):
    fz = (1 + f) / (1 - f)
    if fit_wt is None:
        wtz = np.power(np.abs(np.power(1 - f, 2)) / (1 - np.abs(np.power(f, 2))), 2)
        wty = np.power(np.abs(np.power(1 + f, 2)) / (1 - np.abs(np.power(f, 2))), 2)
        # wt = np.power(1 / (1 - np.abs(np.power(f, 2))), 1)
        # wtz = np.ones(len(s))
        # wty = np.ones(len(s))
    else:
        wtz = fit_wt * np.power(np.abs(np.power(1 - f, 2)), 2)
        wty = fit_wt * np.power(np.abs(np.power(1 + f, 2)), 2)
    # Call vector_fitting function and do some post processing
    if s_dc and s_inf:
        raise RuntimeError('Does not support dc and inf reflection simultaneously!')
    elif s_dc:
        fz0 = np.flip(fz, 0).conj()
        s0 = (1 / np.flip(s, 0)).conj()
        wtz = np.flip(wtz, 0)
        wty = np.flip(wty, 0)
        if s_dc == -1:  # fit z(1/s)
            fz_model = vector_fitting_rescale(fz0, s0, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wtz, bound_wt_z=bound_wt)
            fs = (fz_model.model(s0) - 1) / (fz_model.model(s0) + 1)
            fs = np.flip(fs, 0).conj()
            f_model = fit_s(fs, s, n_pole=n_pole, n_iter=n_iter, s_dc=-1, fit_wt=np.ones(len(s)))
            f_model.stable = fz_model.stable
        elif s_dc == 1:  # fit 1/z(1/s)
            fy_model = vector_fitting_rescale(1/fz0, s0, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wty, bound_wt_z=bound_wt)
            fs = (1 - fy_model.model(s0)) / (1 + fy_model.model(s0))
            fs = np.flip(fs, 0).conj()
            f_model = fit_s(fs, s, n_pole=n_pole, n_iter=n_iter, s_dc=1, fit_wt=np.ones(len(s)))
            f_model.stable = fy_model.stable
        else:  # not supported
            raise RuntimeError('Does not support z_dc not 0 or infinity!')
        # f_model = f_model.inverse_freq()
    elif s_inf:
        if s_inf == -1:  # fit z(s)
            fz_model = vector_fitting_rescale(fz, s, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wtz, bound_wt_z=bound_wt)
            fs = (fz_model.model(s) - 1) / (fz_model.model(s) + 1)
            f_model = fit_s(fs, s, n_pole=n_pole, n_iter=n_iter, s_inf=-1, fit_wt=np.ones(len(s)))
            f_model.stable = fz_model.stable
        elif s_inf == 1:  # fit 1/z(s)
            fy_model = vector_fitting_rescale(1/fz, s, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wty, bound_wt_z=bound_wt)
            fs = (1 - fy_model.model(s)) / (1 + fy_model.model(s))
            f_model = fit_s(fs, s, n_pole=n_pole, n_iter=n_iter, s_inf=1, fit_wt=np.ones(len(s)))
            f_model.stable = fy_model.stable
        else:  # not supported
            raise RuntimeError('Does not support z_inf not 0 or infinity!')
    else:  # no specific reflection point
        f_model = vector_fitting_rescale(f, s, n_pole, n_iter, has_const=True, has_linear=False)

    return f_model


def fit_s(f, s, n_pole=10, n_iter=10, s_dc=None, s_inf=None, fit_wt=None, bound_wt=None):
    if fit_wt is None:
        wt = np.power(1 / (1 - np.abs(np.power(f, 2))), 1)
        # wt = np.ones(len(s))
    else:
        wt = fit_wt
    # Call vector_fitting function and do some post processing
    if s_dc and s_inf:
        raise RuntimeError('Does not support dc and inf reflection simultaneously!')
    elif s_dc:
        # if s_dc == 1:  # 1-s is bounded
        #     f_model = vector_fitting_rescale(1-f, s, n_pole, n_iter, has_const=True, has_linear=False, reflect_z=0)
        #     f_model.const = 1 - f_model.const
        #     f_model.residue = -f_model.residue
        # elif s_dc == -1:  # 1+s is bounded
        #     f_model = vector_fitting_rescale(1+f, s, n_pole, n_iter, has_const=True, has_linear=False, reflect_z=0)
        #     f_model.const -= 1
        # else:  # not supported
        #     raise RuntimeError('Does not support s_dc not +/-1!')

        # New method for fitting s_dc: inverse frequency
        f0 = np.flip(f, 0).conj()
        s0 = (1 / np.flip(s, 0)).conj()
        wt = np.flip(wt, 0)
        if s_dc == 1:  # 1-s is bounded
            f_model = vector_fitting_rescale(1-f0, s0, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wt, bound_wt=bound_wt)
            f_model.const = 1
            f_model.residue = -f_model.residue
        elif s_dc == -1:  # 1+s is bounded
            f_model = vector_fitting_rescale(1+f0, s0, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wt, bound_wt=bound_wt)
            f_model.const = -1
        else:  # not supported
            raise RuntimeError('Does not support s_dc not +/-1!')
        f_model = f_model.inverse_freq()
    elif s_inf:
        if s_inf == 1:  # 1-s is bounded
            f_model = vector_fitting_rescale(1-f, s, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wt, bound_wt=bound_wt)
            f_model.const = 1
            f_model.residue = -f_model.residue
        elif s_inf == -1:  # 1+s is bounded
            f_model = vector_fitting_rescale(1+f, s, n_pole, n_iter, has_const=False, has_linear=False, fit_wt=wt, bound_wt=bound_wt)
            f_model.const = -1
        else:  # not supported
            raise RuntimeError('Does not support s_inf not +/-1!')
    else:  # no specific reflection point
        f_model = vector_fitting_rescale(f, s, n_pole, n_iter, has_const=True, has_linear=False)

    return f_model


def bound_tightening(f, s, s0=None, fs0=None, err=-20):
    n_pole_max = 50
    error_limit = 10 ** (err / 20)  # -20 dB error limit
    norm_order = np.inf
    n_iter = 20  # Number of iterations for fitting
    wt_iter = 20  # Number of iterations for weight updating
    init_wt = 0.01  # Weight updating initial step

    s_all = np.logspace(np.log10(np.min(np.abs(s))) - 3, np.log10(np.max(np.abs(s))) + 3, int(1e5)) * 1j
    f_out = None

    reflect_list = [(-1, None), (1, None), (None, -1), (None, 1)]
    if s0 == 0:
        if fs0 == 1 or fs0 == -1:
            reflect_list = [(fs0, None)]
        else:
            reflect_list = [(-1, None), (1, None)]
    elif s0 is not None and np.isinf(s0):
        if fs0 == 1 or fs0 == -1:
            reflect_list = [(None, fs0)]
        else:
            reflect_list = [(None, -1), (None, 1)]

    for n_pole in range(1, n_pole_max+1):
        for s_dc, s_inf in reflect_list:
            # initial wt is 0
            wt = 0.0
            wt_a = 0.0
            wt_b = 0.0
            start_tighten = False
            backtrace = False
            for i in range(wt_iter):
                f_model = fit_s(f, s, n_pole, n_iter, s_dc, s_inf, bound_wt=wt)

                # check if the result is passive
                small_error = np.linalg.norm(f_model.model(s) - f, norm_order) < error_limit
                passive = np.all(np.abs(f_model.model(s_all)) <= 1)
                valid = small_error and passive
                if valid:
                    f_out = copy.copy(f_model)

                if valid and f_model.stable and not start_tighten:
                    start_tighten = True
                    wt = init_wt
                elif start_tighten:
                    if valid and f_model.stable:
                        wt_a = wt
                        if backtrace:
                            wt = (wt_a + wt_b) / 2
                        else:
                            wt *= 2
                    else:
                        backtrace = True
                        wt_b = wt
                        wt = (wt_a + wt_b) / 2
                else:  # not valid: change reflection or increase number of poles
                    break
            else:
                return f_out
    return f_out


def bound_tightening_sweep(f, s, s0=None, fs0=None, fit_wt=None):
    n_pole_max = 50
    n_iter = 20  # Number of iterations for fitting
    wt_iter = 20  # Number of iterations for weight updating
    init_wt = 0.01  # Weight updating initial step

    s_all = np.logspace(np.log10(np.min(np.abs(s))) - 3, np.log10(np.max(np.abs(s))) + 3, int(1e5)) * 1j
    f_out = None
    param = []

    reflect_list = [(-1, None), (1, None), (None, -1), (None, 1)]
    if s0 == 0:
        if fs0 == 1 or fs0 == -1:
            reflect_list = [(fs0, None)]
        else:
            reflect_list = [(-1, None), (1, None)]
    elif s0 is not None and np.isinf(s0):
        if fs0 == 1 or fs0 == -1:
            reflect_list = [(None, fs0)]
        else:
            reflect_list = [(None, -1), (None, 1)]

    f_inf = []
    b_inf = []
    db_inf = []
    wt_inf = []
    pole_num_inf = []
    f_zero = []
    b_zero = []
    db_zero = []
    wt_zero = []
    pole_num_zero = []

    for n_pole in range(1, n_pole_max+1):
        for s_dc, s_inf in reflect_list:
            # initial wt is 0
            wt = 0.0
            wt_a = 0.0
            wt_b = 0.0
            start_tighten = False
            backtrace = False
            reflect = 0 if s_inf is None else np.inf
            fl = []
            bl = []
            dbl = []
            wtl = []
            for i in range(wt_iter):
                f_model = fit_s_v2(f, s, n_pole, n_iter, s_dc, s_inf, bound_wt=wt, fit_wt=fit_wt)

                # check if the result is passive
                passive = np.all(np.abs(f_model.model(s_all)) <= 1)
                valid = passive
                if valid:
                    # f_out = copy.copy(f_model)
                    # calculate the B + delta B to be used as the stopping condition
                    bound, bw = f_model.bound(reflect)
                    bound_error = f_model.bound_error(f, s, reflect=reflect)
                    fl.append(f_model)
                    bl.append(bound)
                    dbl.append(bound_error)
                    wtl.append(wt)

                # search for the weight
                if valid and f_model.stable and not start_tighten:
                    start_tighten = True
                    wt = init_wt
                elif start_tighten:
                    if valid and f_model.stable:
                        wt_a = wt
                        if backtrace:
                            wt = (wt_a + wt_b) / 2
                        else:
                            wt *= 2
                    else:
                        backtrace = True
                        wt_b = wt
                        wt = (wt_a + wt_b) / 2
                else:  # not valid: change reflection or increase number of poles
                    break
            if reflect == 0 and fl:
                f_zero.append(fl)
                b_zero.append(bl)
                db_zero.append(dbl)
                wt_zero.append(wtl)
                pole_num_zero.append(n_pole)
            elif fl:  # reflect at inf
                f_inf.append(fl)
                b_inf.append(bl)
                db_inf.append(dbl)
                wt_inf.append(wtl)
                pole_num_inf.append(n_pole)
        # check the exit condition here for pole loop
        # exit condition: if the B + dB for the current loop of n_pole is at most n-th smallest
        n = 5
        exit_cond_zero = False
        exit_cond_inf = False
        if len(f_zero) > n:
            b_min = []
            for i in range(len(f_zero)):
                b_min.append(np.min(np.array(b_zero[i]) + np.array(db_zero[i])))
            b_min = np.array(b_min)
            idx = [i for i, p in enumerate(pole_num_zero) if p == n_pole]
            exit_cond_zero = True
            for i in idx:
                if np.sum(b_min < b_min[i]) < n:
                    exit_cond_zero = False
        elif len(f_inf) > n:
            b_min = []
            for i in range(len(f_inf)):
                b_min.append(np.min(np.array(b_inf[i]) + np.array(db_inf[i])))
            b_min = np.array(b_min)
            idx = [i for i, p in enumerate(pole_num_inf) if p == n_pole]
            exit_cond_inf = True
            for i in idx:
                if np.sum(b_min < b_min[i]) < n:  # FIXME: something wrong with the exit condition
                    exit_cond_inf = False
        if exit_cond_zero or exit_cond_inf:
            break

    # return the result here
    # get the function index with minimum B + dB for both zero and inf
    f_out_zero = None
    f_out_inf = None
    idx_zero = []
    idx_inf = []
    b_min_zero = np.inf
    b_min_inf = np.inf
    for i, fl in enumerate(f_zero):
        for j, f in enumerate(fl):
            if b_zero[i][j] + db_zero[i][j] < b_min_zero:
                b_min_zero = b_zero[i][j] + db_zero[i][j]
                idx_zero = [i, j]
                f_out_zero = copy.copy(f)
    for i, fl in enumerate(f_inf):
        for j, f in enumerate(fl):
            if b_inf[i][j] + db_inf[i][j] < b_min_inf:
                b_min_inf = b_inf[i][j] + db_inf[i][j]
                idx_inf = [i, j]
                f_out_inf = copy.copy(f)

    # pick f_out_zero or f_out_inf?
    # choose the one with smaller number of poles
    if not pole_num_inf or (pole_num_zero and pole_num_inf and pole_num_zero[idx_zero[0]] < pole_num_inf[idx_inf[0]]):
        f_out = f_out_zero
        param = [b_zero, db_zero, wt_zero, pole_num_zero, idx_zero]
    elif not pole_num_zero or (pole_num_inf and pole_num_zero and pole_num_zero[idx_zero[0]] >= pole_num_inf[idx_inf[0]]):
        f_out = f_out_inf
        param = [b_inf, db_inf, wt_inf, pole_num_inf, idx_inf]
    return f_out, param


def mode_fitting(f, s, bound_output=False):
    wt = 1e3 * len(s)
    N = np.size(f, 0)
    A = np.concatenate([f, np.reshape(wt * np.identity(N), [N, N, 1])], 2)
    U, Sigma, Vh, dA, err_norm, orig_norm = joint_svd(A)
    all_model = []
    pole = np.zeros(0, dtype=np.complex128)
    residue = np.zeros([N, N, 0], dtype=np.complex128)
    const = np.zeros([N, N], dtype=np.complex128)

    # calculate the singular values of each frequency to get the fit weight for the matrix
    fit_wt = np.ones(len(s))
    for i in range(len(s)):
        # u, s, vh = np.linalg.svd(f[:, :, i])
        sv = np.linalg.svd(f[:, :, i], compute_uv=False)
        fit_wt[i] = np.sqrt(1 + (np.power(sv[0], 2) - np.power(sv[-1], 2)) / np.power(1 - sv[0], 2)) / (1 - np.power(sv[0], 2))

    for i in range(N):
        sigma_n = Sigma[i, i, :-1]
        # sigma_model = bound_tightening(sigma_n, s, np.inf)
        sigma_model, log = bound_tightening_sweep(sigma_n, s, np.inf, fit_wt=fit_wt)
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


