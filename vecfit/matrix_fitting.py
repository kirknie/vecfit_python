# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:14:44 2017

@author: kirknie
"""

import numpy as np
from .rational_fct import RationalMtx, RationalRankOneMtx, vec2mat, mat2vec
from .vector_fitting import pair_pole, init_pole, calculate_zero, lstsq
from .gradient_descent import gradient_descent


def matrix_fitting(f, s, n_pole=10, n_iter=10, has_const=True, has_linear=True, fixed_pole=None):
    w = np.imag(s)
    if fixed_pole is not None:
        n_fixed = len(fixed_pole)
        if n_fixed > n_pole:
            raise RuntimeError('Too many fixed poles!')
        p = np.concatenate([fixed_pole, init_pole(w[-1], n_pole - n_fixed)])
    else:
        p = init_pole(w[-1], n_pole)
    fk = RationalMtx(p, np.zeros([np.size(f, 0), np.size(f, 1), np.size(p)], dtype=np.complex128))  # need to be updated before use

    for k in range(n_iter):
        fk = iteration_step(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=fixed_pole)
    f_model = final_step(f, s, fk, has_const=has_const, has_linear=has_linear)

    return f_model


def matrix_fitting_rescale(f, s, *args, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[0, 0, -1])

    if 'fixed_pole' in kwargs and kwargs['fixed_pole'] is not None:
        kwargs['fixed_pole'] = [p/s_scale for p in kwargs['fixed_pole']]
    if 'reflect_z' in kwargs and kwargs['reflect_z'] is not None:
        kwargs['reflect_z'] /= s_scale

    f_model = matrix_fitting(f/f_scale, s/s_scale, *args, **kwargs)
    f_model.pole *= s_scale
    f_model.residue *= f_scale * s_scale
    if f_model.const is not None:
        f_model.const *= f_scale
    if f_model.linear is not None:
        f_model.linear *= f_scale / s_scale

    return f_model


def iteration_step(f, s, fk, has_const, has_linear, fixed_pole):
    n_pole = len(fk.pole)
    n_freq = len(s)
    f_vec = mat2vec(f)
    n_vec = np.size(f_vec, 0)
    n_fixed = 0
    if fixed_pole is not None:
        n_fixed = len(fixed_pole)
        fk.pole[:n_fixed] = fixed_pole  # should not need this, fixed poles already in there
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h, q_i]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0

    # Construct A
    A = np.zeros((n_freq * n_vec, (n_pole + col_d + col_h) * n_vec + n_pole - n_fixed), dtype=np.complex128)
    A1 = np.zeros((n_freq, n_pole + col_d + col_h), dtype=A.dtype)  # for r, d, h, not q

    for i, p in enumerate(fk.pole):
        if pole_pair[i] == 0:
            A1[:, i] = 1 / (s - p)
        elif pole_pair[i] == 1:
            A1[:, i] = 1 / (s - p) + 1 / (s - p.conj())
            A1[:, i+1] = 1j / (s - p) - 1j / (s - p.conj())
    if has_const:
        A1[:, n_pole] = 1
    if has_linear:
        A1[:, n_pole + col_d] = s
    for i in range(n_vec):
        A[np.size(A1, 0)*i:np.size(A1, 0)*(i+1), np.size(A1, 1)*i:np.size(A1, 1)*(i+1)] = A1[:, :]
        for j in range(n_fixed, n_pole):
            A[np.size(A1, 0)*i:np.size(A1, 0)*(i+1), -n_pole+j] = -A1[:, j] * f_vec[i, :]

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    b = f_vec.reshape(n_freq * n_vec)  # order in rows (nF)
    b = np.concatenate([np.real(b), np.imag(b)])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    rk = np.zeros([n_vec, n_pole], dtype=np.complex128)
    for i in range(n_vec):
        rk[i, :] = x[np.size(A1, 1)*i:np.size(A1, 1)*i+n_pole]
    if n_fixed == n_pole:
        qk = np.array([])
    else:
        qk = np.complex128(x[-n_pole + n_fixed:])
    for i, pp in enumerate(pole_pair):
        if pp == 1:
            r1 = np.copy(rk[:, i])
            r2 = np.copy(rk[:, i+1])
            rk[:, i] = r1 + 1j*r2
            rk[:, i+1] = r1 - 1j*r2
            if i >= n_fixed:
                q1, q2 = qk[i-n_fixed:i-n_fixed+2]
                qk[i-n_fixed] = q1 + 1j * q2
                qk[i-n_fixed + 1] = q1 - 1j * q2
    dk = x[n_pole::np.size(A1, 1)] if col_d else None
    hk = x[n_pole+col_d::np.size(A1, 1)] if col_h else None
    pk = calculate_zero(fk.pole[n_fixed:], qk, 1)
    if fixed_pole is not None:
        pk = np.concatenate([fixed_pole, pk])
    unstable = np.real(pk) > 0
    pk[unstable] -= 2*np.real(pk)[unstable]

    # Convert the vector back to matrix
    return RationalMtx(pk, vec2mat(rk), vec2mat(dk), vec2mat(hk))


def final_step(f, s, fk, has_const, has_linear):
    n_pole = len(fk.pole)
    n_freq = len(s)
    f_vec = mat2vec(f)
    n_vec = np.size(f_vec, 0)
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h, q_i]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0

    # Construct A
    A = np.zeros((n_freq * n_vec, (n_pole + col_d + col_h) * n_vec), dtype=np.complex128)
    A1 = np.zeros((n_freq, n_pole + col_d + col_h), dtype=A.dtype)  # for r, d, h, not q

    for i, p in enumerate(fk.pole):
        if pole_pair[i] == 0:
            A1[:, i] = 1 / (s - p)
        elif pole_pair[i] == 1:
            A1[:, i] = 1 / (s - p) + 1 / (s - p.conj())
            A1[:, i+1] = 1j / (s - p) - 1j / (s - p.conj())
    if has_const:
        A1[:, n_pole] = 1
    if has_linear:
        A1[:, n_pole + col_d] = s
    for i in range(n_vec):
        A[np.size(A1, 0)*i:np.size(A1, 0)*(i+1), np.size(A1, 1)*i:np.size(A1, 1)*(i+1)] = A1[:, :]

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    b = f_vec.reshape(n_freq * n_vec)  # order in rows (nF)
    b = np.concatenate([np.real(b), np.imag(b)])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    rk = np.zeros([n_vec, n_pole], dtype=np.complex128)
    for i in range(n_vec):
        rk[i, :] = x[np.size(A1, 1)*i:np.size(A1, 1)*i+n_pole]
    for i, pp in enumerate(pole_pair):
        if pp == 1:
            r1 = np.copy(rk[:, i])
            r2 = np.copy(rk[:, i+1])
            rk[:, i] = r1 + 1j*r2
            rk[:, i+1] = r1 - 1j*r2
    dk = x[n_pole::np.size(A1, 1)] if col_d else None
    hk = x[n_pole+col_d::np.size(A1, 1)] if col_h else None

    # Convert the vector back to matrix
    return RationalMtx(fk.pole, vec2mat(rk), vec2mat(dk), vec2mat(hk))


def matrix_fitting_rank_one(f, s, n_pole=10, n_iter=10, has_const=True, has_linear=True, fixed_pole=None):
    fk = matrix_fitting(f, s, n_pole, n_iter, has_const, has_linear, fixed_pole)
    fk = fk.rank_one()
    for k in range(n_iter):
        fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=fk.pole, update_residue='left', update_pole=False)
        fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=fk.pole, update_residue='right', update_pole=False)
    # for k in range(n_iter):
    #     fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update_residue='left', update_pole=False)
    #     fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update_residue='right', update_pole=False)
    # fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update_residue='right', update_pole=True)

    # fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update='right')
    # for k in range(n_iter):
    #     fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update='left')
    #     fk = iteration_rank_one(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=None, update='right')
    
    # # gradient descent on poles
    # # x = [real pole, complex pole real part, complex pole imag part]
    # pole_pair = pair_pole(fk.pole)
    # x0 = np.zeros(fk.pole.shape)
    # for i, pp in enumerate(pole_pair):
    #     if pp == 0:
    #         x0[i] = fk.pole[i].real  # no imag part
    #     elif pp == 1:
    #         x0[i] = fk.pole[i].real
    #         x0[i+1] = fk.pole[i].imag
    # y = lambda x: matrix_cost_fct(x, fk, f, s)
    # dy = lambda x: d_matrix_cost_fct(x, fk, f, s)
    # x, x_list = gradient_descent(y, dy, x0, return_step=True)
    #
    # pole = np.zeros(fk.pole.shape, dtype=np.complex128)
    # for i, pp in enumerate(pole_pair):
    #     if pp == 0:
    #         pole[i] = x[i]
    #     elif pp == 1:
    #         pole[i] = x[i] + 1j*x[i+1]
    #         pole[i+1] = x[i] - 1j*x[i+1]
    # fk.pole = pole

    return fk


def matrix_cost_fct(x, model, f, s):
    pole_pair = pair_pole(model.pole)
    y = 0
    for i1 in range(np.size(f, 0)):
        for i2 in range(np.size(f, 1)):
            d = model.const[i1, i2].real
            h = model.linear[i1, i2].real
            for i3 in range(np.size(f, 2)):
                w = s[i3].imag
                ele = f[i1, i2, i3]
                er = 0  # error real part
                ei = 0  # error imag part
                for i, pp in enumerate(pole_pair):
                    residue = np.outer(model.residue_left[:, i], model.residue_right[:, i])
                    r = residue[i1, i2]
                    if pp == 0:
                        xi = x[i]
                        # steps of calculating yi
                        a1 = -r.real * xi
                        a2 = xi**2
                        a3 = a2 + w**2
                        a4 = a1 / a3
                        er += a4
                        a5 = -w * r.real / a3
                        ei += a5
                    elif pp == 1:
                        xi_real = x[i]
                        xi_imag = x[i + 1]
                        # first for er
                        a1 = -r.real * xi_real
                        a2 = xi_imag - w
                        a3 = r.imag * a2
                        a4 = a1 - a3
                        a5 = xi_real**2
                        a6 = a2**2
                        a7 = a5 + a6
                        a8 = a4 / a7
                        a9 = xi_imag + w
                        a10 = r.imag * a9
                        a11 = a1 - a10
                        a12 = a9**2
                        a13 = a5 + a12
                        a14 = a11 / a13
                        a15 = a8 + a14
                        er += a15
                        # then for ei
                        a1 = -r.imag * xi_real
                        a2 = xi_imag - w
                        a3 = r.real * a2
                        a4 = a1 + a3
                        a5 = xi_real**2
                        a6 = a2**2
                        a7 = a5 + a6
                        a8 = a4 / a7
                        a9 = xi_imag + w
                        a10 = r.real * a9
                        a11 = -a1 - a10
                        a12 = a9**2
                        a13 = a5 + a12
                        a14 = a11 / a13
                        a15 = a8 + a14
                        ei += a15
                er += d - ele.real
                ei += h * w - ele.imag
                e = er**2 + ei**2
                y += e
    return y


def matrix_cost_fct2(x, model, f, s):
    pole_pair = pair_pole(model.pole)
    pole = np.zeros(model.pole.shape, dtype=np.complex128)
    for i, pp in enumerate(pole_pair):
        if pp == 0:
            pole[i] = x[i]
        elif pp == 1:
            pole[i] = x[i] + 1j*x[i+1]
            pole[i+1] = x[i] - 1j*x[i+1]
    tmp_model = RationalRankOneMtx(pole, model.residue_left, model.residue_right, model.const, model.linear)
    error = tmp_model.model(s) - f
    y = np.sum(np.abs(error)**2)
    return y


def d_matrix_cost_fct(x, model, f, s):
    pole_pair = pair_pole(model.pole)
    y = 0
    dy = np.zeros(x.shape)
    for i1 in range(np.size(f, 0)):
        for i2 in range(np.size(f, 1)):
            d = model.const[i1, i2].real
            h = model.linear[i1, i2].real
            for i3 in range(np.size(f, 2)):
                w = s[i3].imag
                ele = f[i1, i2, i3]
                er = 0  # error real part
                ei = 0  # error imag part
                der = np.zeros(x.shape)
                dei = np.zeros(x.shape)
                for i, pp in enumerate(pole_pair):
                    residue = np.outer(model.residue_left[:, i], model.residue_right[:, i])
                    r = residue[i1, i2]
                    if pp == 0:
                        xi = x[i]
                        dxi = np.zeros(x.shape)
                        dxi[i] = 1
                        # steps of calculating yi
                        a1 = -r.real * xi
                        da1 = -r.real * dxi
                        a2 = xi**2
                        da2 = 2 * xi * dxi
                        a3 = a2 + w**2
                        da3 = da2
                        a4 = a1 / a3
                        da4 = da1 / a3 - a1 / a3**2 * da3
                        er += a4
                        der += da4
                        a5 = -w * r.real / a3
                        da5 = w * r.real / a3**2 * da3
                        ei += a5
                        dei += da5
                    elif pp == 1:
                        xi_real = x[i]
                        xi_imag = x[i + 1]
                        dxi_real = np.zeros(x.shape)
                        dxi_real[i] = 1
                        dxi_imag = np.zeros(x.shape)
                        dxi_imag[i+1] = 1
                        # first for er
                        a1 = -r.real * xi_real
                        da1 = -r.real * dxi_real
                        a2 = xi_imag - w
                        da2 = dxi_imag
                        a3 = r.imag * a2
                        da3 = r.imag * da2
                        a4 = a1 - a3
                        da4 = da1 - da3
                        a5 = xi_real**2
                        da5 = 2 * xi_real * dxi_real
                        a6 = a2**2
                        da6 = 2 * a2 * da2
                        a7 = a5 + a6
                        da7 = da5 + da6
                        a8 = a4 / a7
                        da8 = da4 / a7 - a4 / a7**2 * da7
                        a9 = xi_imag + w
                        da9 = dxi_imag
                        a10 = r.imag * a9
                        da10 = r.imag * da9
                        a11 = a1 - a10
                        da11 = da1 - da10
                        a12 = a9**2
                        da12 = 2 * a9 * da9
                        a13 = a5 + a12
                        da13 = da5 + da12
                        a14 = a11 / a13
                        da14 = da11 / a13 - a11 / a13**2 * da13
                        a15 = a8 + a14
                        da15 = da8 + da14
                        er += a15
                        der += da15
                        # then for ei
                        a1 = -r.imag * xi_real
                        da1 = -r.imag * dxi_real
                        a2 = xi_imag - w
                        da2 = dxi_imag
                        a3 = r.real * a2
                        da3 = r.real * da2
                        a4 = a1 + a3
                        da4 = da1 + da3
                        a5 = xi_real**2
                        da5 = 2 * xi_real * dxi_real
                        a6 = a2**2
                        da6 = 2 * a2 * da2
                        a7 = a5 + a6
                        da7 = da5 + da6
                        a8 = a4 / a7
                        da8 = da4 / a7 - a4 / a7**2 * da7
                        a9 = xi_imag + w
                        da9 = dxi_imag
                        a10 = r.real * a9
                        da10 = r.real * da9
                        a11 = -a1 - a10
                        da11 = -da1 - da10
                        a12 = a9**2
                        da12 = 2 * a9 * da9
                        a13 = a5 + a12
                        da13 = da5 + da12
                        a14 = a11 / a13
                        da14 = da11 / a13 - a11 / a13**2 * da13
                        a15 = a8 + a14
                        da15 = da8 + da14
                        ei += a15
                        dei += da15
                er += d - ele.real
                ei += h * w - ele.imag
                e = er**2 + ei**2
                de = 2 * er * der + 2 * ei * dei
                y += e
                dy += de
    return dy


def d_matrix_cost_fct2(x, model, f, s):
    dy = np.zeros(x.shape)
    e = 1e-10
    for i, dyi in enumerate(dy):
        p = np.zeros(x.shape)
        p[i] = e
        dy[i] = (matrix_cost_fct(x + p, model, f, s) - matrix_cost_fct(x - p, model, f, s)) / (2 * e)
    return dy


def matrix_fitting_rank_one_rescale(f, s, *args, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[0, 0, -1])

    if 'fixed_pole' in kwargs and kwargs['fixed_pole'] is not None:
        kwargs['fixed_pole'] = [p/s_scale for p in kwargs['fixed_pole']]
    if 'reflect_z' in kwargs and kwargs['reflect_z'] is not None:
        kwargs['reflect_z'] /= s_scale

    f_model = matrix_fitting_rank_one(f/f_scale, s/s_scale, *args, **kwargs)
    f_model.pole *= s_scale
    f_model.residue_left *= np.sqrt(f_scale * s_scale)
    f_model.residue_right *= np.sqrt(f_scale * s_scale)
    if f_model.const is not None:
        f_model.const *= f_scale
    if f_model.linear is not None:
        f_model.linear *= f_scale / s_scale

    return f_model


def iteration_rank_one(f, s, fk, has_const, has_linear, fixed_pole, update_residue, update_pole):
    if update_residue != 'left' and update_residue != 'right' and update_residue is not None:
        raise RuntimeError('The update_residue must be set to left/right/None!')

    n_pole = len(fk.pole)
    n_freq = len(s)
    n_mat = np.size(f, 0)
    f_vec = mat2vec(f, False)  # non-symmetric matrix
    n_vec = np.size(f_vec, 0)
    n_fixed = 0
    if fixed_pole is not None:
        n_fixed = len(fixed_pole)
        fk.pole[:n_fixed] = fixed_pole  # should not need this, fixed poles already in there
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h, q_i]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0

    # Construct A
    # rows: element 00 all freq, element 01 all freq, ...
    # cols: element 0 of all poles, element 1 of all poles, const 00, const 01, ..., linear 00, linear 01, ...
    if update_residue is None:
        A = np.zeros((n_freq * n_vec, n_pole - n_fixed), dtype=np.complex128)
    else:
        A = np.zeros((n_freq * n_vec, n_pole * n_mat + (col_d + col_h) * n_vec + n_pole - n_fixed), dtype=np.complex128)
        # A1 = np.zeros((n_freq, n_pole + col_d + col_h), dtype=A.dtype)  # for r, d, h, not q

        idx = 0
        for i in range(n_mat):
            for j in range(n_mat):
                # Fill in the corresponding rows of A, constraints on S_ij
                row_range = range(idx*n_freq, (idx+1)*n_freq)
                for k, p in enumerate(fk.pole):
                    if pole_pair[k] == 0:
                        if update_residue == 'left':
                            A[row_range, i * n_pole + k] = fk.residue_right[j, k] / (s - p)
                        else:
                            A[row_range, j * n_pole + k] = fk.residue_left[i, k] / (s - p)
                    if pole_pair[k] == 1:
                        if update_residue == 'left':
                            A[row_range, i * n_pole + k] = fk.residue_right[j, k] / (s - p) + fk.residue_right[j, k].conj() / (s - p.conj())
                            A[row_range, i * n_pole + k + 1] = 1j * fk.residue_right[j, k] / (s - p) - 1j * fk.residue_right[j, k].conj() / (s - p.conj())
                        else:
                            A[row_range, j * n_pole + k] = fk.residue_left[i, k] / (s - p) + fk.residue_left[i, k].conj() / (s - p.conj())
                            A[row_range, j * n_pole + k + 1] = 1j * fk.residue_left[i, k] / (s - p) - 1j * fk.residue_left[i, k].conj() / (s - p.conj())
                if has_const:
                    A[row_range, n_pole * n_mat + idx] = 1
                if has_linear:
                    A[row_range, n_pole * n_mat + col_d * n_vec + idx] = s
                idx += 1

    for i in range(n_vec):
        row_range = range(i*n_freq, (i+1)*n_freq)
        for j in range(n_fixed, n_pole):
            p = fk.pole[j]
            if pole_pair[j] == 0:
                A[row_range, -n_pole+j] = -f_vec[i, :] / (s - p)
            elif pole_pair[j] == 1:
                A[row_range, -n_pole+j] = -f_vec[i, :] / (s - p) - f_vec[i, :] / (s - p.conj())
                A[row_range, -n_pole+j+1] = -1j * f_vec[i, :] / (s - p) - 1j * f_vec[i, :] / (s - p.conj())

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    # b: element 00 all freq, element 01 all freq, ...
    b = f_vec.reshape(n_freq * n_vec)  # order in rows (nF)
    b = np.concatenate([np.real(b), np.imag(b)])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    # x: element 0 of all poles, element 1 of all poles, const 00, const 01, ..., linear 00, linear 01, ...
    if update_residue is not None:
        rk = np.zeros([n_mat, n_pole], dtype=np.complex128)
        for i in range(n_mat):
            rk[i, :] = x[n_pole*i:n_pole*(i+1)]
        for i, pp in enumerate(pole_pair):
            if pp == 1:
                r1 = np.copy(rk[:, i])
                r2 = np.copy(rk[:, i + 1])
                rk[:, i] = r1 + 1j * r2
                rk[:, i + 1] = r1 - 1j * r2
        if col_d is None:
            dk = None
        else:
            dk = x[n_pole * n_mat:n_pole * n_mat + n_vec]
            dk = vec2mat(dk, False)
        if col_h is None:
            hk = None
        else:
            hk = x[n_pole * n_mat + n_vec * col_d:n_pole * n_mat + n_vec * col_d + n_vec]
            hk = vec2mat(hk, False)

    if update_pole is True:
        if n_fixed == n_pole:
            qk = np.array([])
        else:
            qk = np.complex128(x[-n_pole + n_fixed:])
        for i, pp in enumerate(pole_pair):
            if pp == 1:
                if i >= n_fixed:
                    q1, q2 = qk[i-n_fixed:i-n_fixed+2]
                    qk[i-n_fixed] = q1 + 1j * q2
                    qk[i-n_fixed + 1] = q1 - 1j * q2
        pk = calculate_zero(fk.pole[n_fixed:], qk, 1)
        if fixed_pole is not None:
            pk = np.concatenate([fixed_pole, pk])
        unstable = np.real(pk) > 0
        if np.any(unstable):
            print('Unstable poles!')
        pk[unstable] -= 2*np.real(pk)[unstable]
    else:
        pk = fk.pole

    # Convert the vector back to matrix
    if update_residue is None:
        return RationalRankOneMtx(pk, fk.residue_left, fk.residue_right, fk.const, fk.linear)
    elif update_residue == 'left':
        return RationalRankOneMtx(pk, rk, fk.residue_right, dk, hk)
    else:
        return RationalRankOneMtx(pk, fk.residue_left, rk, dk, hk)


def iteration_symmetric_rank_one(f, s, fk, has_const, has_linear, fixed_pole, update):
    n_pole = len(fk.pole)
    n_freq = len(s)
    n_mat = np.size(f, 0)
    f_vec = mat2vec(f)
    n_vec = np.size(f_vec, 0)
    n_fixed = 0
    if fixed_pole is not None:
        n_fixed = len(fixed_pole)
        fk.pole[:n_fixed] = fixed_pole  # should not need this, fixed poles already in there
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h, q_i]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0

    real_residue = []
    for i, pp in enumerate(pole_pair):
        if pp == 0 and np.all(np.abs(fk.residue[:, i].imag) < 1e-8 * np.abs(fk.residue[:, i].real)):
            real_residue.append(True)
        else:
            real_residue.append(False)

    # Construct A
    # rows: element 00 all freq, element 01 all freq, ...
    # cols: element 0 of all poles, element 1 of all poles, const 00, const 01, ..., linear 00, linear 01, ...
    A = np.zeros((n_freq * n_vec, n_pole * n_mat + (col_d + col_h) * n_vec + n_pole - n_fixed), dtype=np.complex128)
    # A1 = np.zeros((n_freq, n_pole + col_d + col_h), dtype=A.dtype)  # for r, d, h, not q

    idx = 0
    for i in range(n_mat):
        for j in range(i, n_mat):
            # Fill in the corresponding rows of A, constraints on S_ij
            row_range = range(idx*n_freq, (idx+1)*n_freq)
            for k, p in enumerate(fk.pole):
                if pole_pair[k] == 0:
                    if real_residue[k]:  # vector is pure real
                        # do r_i*r_j_new/2 + r_j*r_i_new/2
                        A[row_range, i*n_pole + k] += fk.residue[j, k] / 2 / (s - p)
                        A[row_range, j*n_pole + k] += fk.residue[i, k] / 2 / (s - p)
                    else:  # vector is pure imag
                        A[row_range, i*n_pole + k] += 1j * fk.residue[j, k] / 2 / (s - p)
                        A[row_range, j*n_pole + k] += 1j * fk.residue[i, k] / 2 / (s - p)
                elif pole_pair[k] == 1:
                    A[row_range, i*n_pole + k] += fk.residue[j, k] / 2 / (s - p) + fk.residue[j, k].conj() / 2 / (s - p.conj())
                    A[row_range, i*n_pole + k + 1] += 1j * fk.residue[j, k] / 2 / (s - p) + 1j * fk.residue[j, k].conj() / 2 / (s - p.conj())
                    A[row_range, j*n_pole + k] += fk.residue[i, k] / 2 / (s - p) - fk.residue[i, k].conj() / 2 / (s - p.conj())
                    A[row_range, j*n_pole + k + 1] += 1j * fk.residue[i, k] / 2 / (s - p) - 1j * fk.residue[i, k].conj() / 2 / (s - p.conj())
            if has_const:
                A[row_range, n_pole * n_mat + idx] += 1
            if has_linear:
                A[row_range, n_pole * n_mat + col_d * n_vec + idx] += s
            idx += 1

    for i in range(n_vec):
        row_range = range(i*n_freq, (i+1)*n_freq)
        for j in range(n_fixed, n_pole):
            p = fk.pole[j]
            if pole_pair[j] == 0:
                A[row_range, -n_pole+j] = -f_vec[i, :] / (s - p)
            elif pole_pair[j] == 1:
                A[row_range, -n_pole+j] = -f_vec[i, :] / (s - p) - f_vec[i, :] / (s - p.conj())
                A[row_range, -n_pole+j+1] = -1j * f_vec[i, :] / (s - p) - 1j * f_vec[i, :] / (s - p.conj())

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    # b: element 00 all freq, element 01 all freq, ...
    b = f_vec.reshape(n_freq * n_vec)  # order in rows (nF)
    b = np.concatenate([np.real(b), np.imag(b)])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    # x: element 0 of all poles, element 1 of all poles, const 00, const 01, ..., linear 00, linear 01, ...
    rk = np.zeros([n_mat, n_pole], dtype=np.complex128)
    for i in range(n_mat):
        rk[i, :] = x[n_pole*i:n_pole*(i+1)]
    if n_fixed == n_pole:
        qk = np.array([])
    else:
        qk = np.complex128(x[-n_pole + n_fixed:])
    for i, pp in enumerate(pole_pair):
        if pp == 1:
            r1 = np.copy(rk[:, i])
            r2 = np.copy(rk[:, i+1])
            rk[:, i] = r1 + 1j*r2
            rk[:, i+1] = r1 - 1j*r2
            if i >= n_fixed:
                q1, q2 = qk[i-n_fixed:i-n_fixed+2]
                qk[i-n_fixed] = q1 + 1j * q2
                qk[i-n_fixed + 1] = q1 - 1j * q2
        elif pp == 0 and not real_residue[i]:
            rk[:, i] *= 1j
    if col_d is None:
        dk = None
    else:
        dk = x[n_pole * n_mat:n_pole * n_mat + n_vec]
        dk = vec2mat(dk)
    # if dk is not None and not real_const:
    #     dk = 1j * dk
    if col_h is None:
        hk = None
    else:
        hk = x[n_pole * n_mat + n_vec * col_d:n_pole * n_mat + n_vec * col_d + n_vec]
        hk = vec2mat(hk)
    # if hk is not None and not real_linear:
    #     hk = 1j * hk
    pk = calculate_zero(fk.pole[n_fixed:], qk, 1)
    if fixed_pole is not None:
        pk = np.concatenate([fixed_pole, pk])
    unstable = np.real(pk) > 0
    pk[unstable] -= 2*np.real(pk)[unstable]

    # Convert the vector back to matrix
    return RationalRankOneMtx(pk, rk, dk, hk)



