# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 15:27:46 2017

@author: kirknie
"""

from .rational_fct import RationalFct
import numpy as np


def init_pole(w, n_pole):
    loss_ratio = 1e-2
    n_pole = int(n_pole)
    
    if n_pole == 1:
        pole = np.array(-loss_ratio*w)
    elif n_pole == 2:
        pole = w * np.array([-loss_ratio-1j, -loss_ratio+1j])
    elif n_pole == 3:
        pole = w * np.array([loss_ratio, -loss_ratio-1j, -loss_ratio+1j])
    elif n_pole > 3 and n_pole%2 == 0:
        complex_pole = np.linspace(0, 1, int((n_pole-2)/2+1))[1:]
        pole = np.concatenate( [[p*(-loss_ratio-1j), p*(-loss_ratio+1j)] for p in complex_pole] )
        pole = w * np.concatenate( [[-1, -10], pole] )
    elif n_pole > 3 and n_pole%2 == 1:
        complex_pole = np.linspace(0, 1, int((n_pole-3)/2+1))[1:]
        pole = np.concatenate( [[p*(-loss_ratio-1j), p*(-loss_ratio+1j)] for p in complex_pole] )
        pole = w * np.concatenate( [[-1, -3, -10], pole] )
    else:
        raise RuntimeError('Error: invalid number of poles')

    return pole


def pair_pole(pole):
    # Function groups pole into complex conjugate pairs
    pole_pair = np.zeros(len(pole))
    
    for i, p in enumerate(pole):
        if p.imag != 0:
            if i == 0 or pole_pair[i-1] != 1:
                if p.conjugate() != pole[i+1]:
                    raise RuntimeError("Complex poles must come in conjugate pairs: %s, %s" % (p, pole[i+1]))
                pole_pair[i] = 1
            else:
                pole_pair[i] = 2
    
    return pole_pair


def calculate_zero(pole, residue, const):
    # First group input poles into complex conjugate pairs
    n_pole = len(pole)
    pole_pair = pair_pole(pole)
    
    # Then calculate the zeros by following Appendix B
    A = np.diag(pole)
    b = np.ones(n_pole)
    c = residue/const
    for i, p in enumerate(pole):
        if pole_pair[i] == 1:
            A[i, i] = A[i+1, i+1] = p.real
            A[i, i+1] = p.imag
            A[i+1, i] = -p.imag
            b[i] = 2
            b[i+1] = 0
            c[i+1] = c[i].imag
            c[i] = c[i].real
    
    H = A - np.outer(b, c)
    # print('H is:', H)
    H = H.real  # H should already be real
    z = np.sort(np.linalg.eigvals(H))  # sorted zeros should have [real, complex pairs]
    return z


def vector_fitting(f, s, n_pole=10, n_iters=10, has_const=True, has_linear=True, fixed_pole=None, reflect_z=None,
                   bound_wt=None):
    if reflect_z is not None and not has_const:
        # has_const = True
        raise RuntimeError('reflect_z is set but has_const is False.  Setting it has_const = True!')

    w = np.imag(s)
    if fixed_pole:
        n_fixed = len(fixed_pole)
        if n_fixed >= n_pole:
            raise RuntimeError('Too many fixed poles!')
        p = np.concatenate([fixed_pole, init_pole(w[-1], n_pole - n_fixed)])
    else:
        p = init_pole(w[-1], n_pole)
    fk = RationalFct(p, residue=None)  # need to be updated before use

    for k in range(n_iters):
        fk = iteration_step(f, s, fk, has_const=has_const, has_linear=has_linear, fixed_pole=fixed_pole,
                            reflect_z=reflect_z, bound_wt=bound_wt)
    f_model = final_step(f, s, fk, has_const=has_const, has_linear=has_linear, reflect_z=reflect_z, bound_wt=bound_wt)

    return f_model


def vector_fitting_rescale(f, s, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])

    f_model = vector_fitting(f/f_scale, s/s_scale, **kwargs)
    f_model.pole *= s_scale
    f_model.residue *= f_scale * s_scale
    f_model.const *= f_scale
    f_model.linear *= f_scale / s_scale

    return f_model


def iteration_step(f, s, fk, has_const, has_linear, fixed_pole, reflect_z, bound_wt):
    n_pole = len(fk.pole)
    n_freq = len(s)
    n_fixed = 0
    if fixed_pole:
        n_fixed = len(fixed_pole)
        fk.pole[:n_fixed] = fixed_pole  # should not need this, fixed poles already in there
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h, q_i]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0
    if reflect_z is not None:
        col_d = 0

    # Construct A
    A = np.zeros((n_freq, 2*n_pole + col_d + col_h - n_fixed), dtype=np.complex128)

    for i, p in enumerate(fk.pole):
        if pole_pair[i] == 0:
            A[:, i] = 1 / (s - p)
        elif pole_pair[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - p.conjugate())
        elif pole_pair[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - p.conjugate())
        else:
            raise RuntimeError("pole_pair[%d] = %d" % (i, pole_pair[i]))
        if i >= n_fixed:
            A[:, -n_pole + i] = -A[:, i] * f
    if has_const:
        A[:, n_pole] = 1
    if has_linear:
        A[:, n_pole + col_d] = s

    if reflect_z is not None:
        s0 = reflect_z
        for i, p in enumerate(fk.pole):
            if pole_pair[i] == 0:
                A[:, i] += -(1 / (s0 - p) + 1 / (-s0 - p)) / 2
            elif pole_pair[i] == 1:
                A[:, i] += -(1 / (s0 - p) + 1 / (-s0 - p)) / 2 - (
                            1 / (s0 - p.conjugate()) + 1 / (-s0 - p.conjugate())) / 2
            elif pole_pair[i] == 2:
                A[:, i] += -1j * (1 / (s0 - p) + 1 / (-s0 - p)) / 2 + 1j * (
                            1 / (s0 - p.conjugate()) + 1 / (-s0 - p.conjugate())) / 2

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    if bound_wt and bound_wt > 0:  # Require fitting S with reflection at infinity, equations same for s_inf = 1 or -1
        a = np.zeros((1, A.shape[1]), dtype=A.dtype)
        for i, p in enumerate(fk.pole):
            if pole_pair[i] == 0:
                a[0, i] = -np.pi/2
            elif pole_pair[i] == 1:
                a[0, i] = -np.pi
            elif pole_pair[i] == 2:
                a[0, i] = 0
            if i >= n_fixed:
                a[0, -n_pole + i] = -2 * a[0, i]
        a *= bound_wt
        A = np.vstack([A, a])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    b = f
    b = np.concatenate([np.real(b), np.imag(b)])
    if bound_wt and bound_wt > 0:
        b = np.concatenate([b, [np.real(np.sum(fk.pole))*np.pi*bound_wt]])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    rk = np.complex128(x[:n_pole])
    qk = np.complex128(x[-n_pole+n_fixed:])
    for i, pp in enumerate(pole_pair):
        if pp == 1:
            r1, r2 = rk[i:i+2]
            rk[i] = r1 - 1j*r2
            rk[i+1] = r1 + 1j*r2
            if i >= n_fixed:
                q1, q2 = qk[i-n_fixed:i-n_fixed+2]
                qk[i] = q1 - 1j * q2
                qk[i + 1] = q1 + 1j * q2
    dk = x[n_pole] if col_d else None
    hk = x[n_pole+col_d] if col_h else None
    pk = calculate_zero(fk.pole[n_fixed:], qk, 1)
    if fixed_pole:
        pk = np.concatenate([fixed_pole, pk])
    unstable = np.real(pk) > 0
    pk[unstable] -= 2*np.real(pk)[unstable]
    if reflect_z is not None:
        dk = np.sum([-(r/(s0-p)+r/(-s0-p))/2 for p, r in zip(fk.pole, rk)])

    return RationalFct(pk, rk, dk, hk)


def final_step(f, s, fk, has_const, has_linear, reflect_z, bound_wt):
    n_pole = len(fk.pole)
    n_freq = len(s)
    pole_pair = pair_pole(fk.pole)

    # Construct the set of linear equations A*x=b
    # x = [r_i, d, h]
    col_d = 1 if has_const else 0
    col_h = 1 if has_linear else 0
    if reflect_z is not None:
        col_d = 0

    # Construct A
    A = np.zeros((n_freq, n_pole + col_d + col_h), dtype=np.complex128)

    for i, p in enumerate(fk.pole):
        if pole_pair[i] == 0:
            A[:, i] = 1 / (s - p)
        elif pole_pair[i] == 1:
            A[:, i] = 1 / (s - p) + 1 / (s - p.conjugate())
        elif pole_pair[i] == 2:
            A[:, i] = 1j / (s - p) - 1j / (s - p.conjugate())
        else:
            raise RuntimeError("pole_pair[%d] = %d" % (i, pole_pair[i]))
    if has_const:
        A[:, n_pole] = 1
    if has_linear:
        A[:, n_pole + col_d] = s

    if reflect_z is not None:
        s0 = reflect_z
        for i, p in enumerate(fk.pole):
            if pole_pair[i] == 0:
                A[:, i] += -(1 / (s0 - p) + 1 / (-s0 - p)) / 2
            elif pole_pair[i] == 1:
                A[:, i] += -(1 / (s0 - p) + 1 / (-s0 - p)) / 2 - (
                            1 / (s0 - p.conjugate()) + 1 / (-s0 - p.conjugate())) / 2
            elif pole_pair[i] == 2:
                A[:, i] += -1j * (1 / (s0 - p) + 1 / (-s0 - p)) / 2 + 1j * (
                            1 / (s0 - p.conjugate()) + 1 / (-s0 - p.conjugate())) / 2

    # Form real equations from complex equations
    A = np.vstack([np.real(A), np.imag(A)])

    if bound_wt and bound_wt > 0:  # Require fitting S with reflection at infinity, equations same for s_inf = 1 or -1
        a = np.zeros((1, A.shape[1]), dtype=A.dtype)
        for i, p in enumerate(fk.pole):
            if pole_pair[i] == 0:
                a[0, i] = -np.pi/2
            elif pole_pair[i] == 1:
                a[0, i] = -np.pi
            elif pole_pair[i] == 2:
                a[0, i] = 0
        a *= bound_wt
        A = np.vstack([A, a])

    cA = np.linalg.cond(A)
    if cA > 1e13:
        print('Warning: Ill Conditioned Matrix.  Cond(A)={:.2e}  Consider scaling the problem down'.format(cA))

    # Construct b
    b = f
    b = np.concatenate([np.real(b), np.imag(b)])
    if bound_wt and bound_wt > 0:
        b = np.concatenate([b, [np.real(np.sum(fk.pole))*np.pi*bound_wt]])

    # Solve for x
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=-1)

    rk = np.complex128(x[:n_pole])
    for i, pp in enumerate(pole_pair):
        if pp == 1:
            r1, r2 = rk[i:i+2]
            rk[i] = r1 - 1j*r2
            rk[i+1] = r1 + 1j*r2
    dk = x[n_pole] if col_d else None
    hk = x[n_pole+col_d] if col_h else None
    if reflect_z is not None:
        dk = np.sum([-(r/(s0-p)+r/(-s0-p))/2 for p, r in zip(fk.pole, rk)])

    return RationalFct(fk.pole, rk, dk, hk)

