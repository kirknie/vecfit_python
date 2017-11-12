#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:42:44 2017

@author: dingnie
"""

import numpy as np
import numpy.linalg
import skrf as rf

def get_z0(snp_file):
    # Input: file name
    # Output: nxl z0
    n = int(snp_file[-2])
    if '.s' + str(n) + 'p' not in snp_file:
        raise RuntimeError('Error: input file is not snp touchstone file')
    z0 = [[] for i in range(n)]
    with open(snp_file) as f:
        for l in f:
            if '! Port Impedance' in l:
                z0_str = list(filter(None, l[len('! Port Impedance'):].split(' ')))
                for i in range(n):
                    z0[i].append(float(z0_str[2*i]) + 1j*float(z0_str[2*i+1]))
    z0 = np.array(z0)
    if n == 1:
        z0 = z0.reshape(len(z0[0]))
    return np.array(z0)


def s2z(s, z0):
    if s.ndim == 1:  # one port
        z = z0 * (1+s) / (1-s)
    elif s.ndim == 3:
        n = s.shape[0]
        l = s.shape[2]
        z = np.zeros(s.shape, dtype=s.dtype)
        for i in range(l):
            z0_sq = np.matrix(np.diag(np.sqrt(z0[:, i])))
            s_ele = s[:, :, i]
            z[:, :, i] = z0_sq * np.matrix(numpy.identity(n)+s_ele) * np.linalg.inv(np.matrix(numpy.identity(n)-s_ele)) * z0_sq
    else:
        raise RuntimeError('Error: unexpected S-parameter format')
    return z


def read_snp(snp_file):
    ant_data = rf.Network(snp_file)
    z0 = get_z0(snp_file)
    n = ant_data.number_of_ports
    freq = ant_data.f
    l = len(freq)
    if n == 1:
        s = ant_data.s.reshape(l)
    else:
        s = np.zeros([n, n, l], dtype=np.complex128)
        for i in range(l):
            s[:, :, i] = ant_data.s[i, :, :]
    z = s2z(s, z0)
    return freq, n, z, s, z0

if __name__ == '__main__':
    s2p_file = 'two_coupled_SIW_ant_39GHz.s2p'
    freq, n, z, s, z0 = read_snp(s2p_file)
    # print(freq, n, z, s, z0)
