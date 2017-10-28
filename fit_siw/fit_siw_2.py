# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:57:57 2017

@author: kirknie
"""

import numpy as np
import matplotlib.pyplot as plt
import snp_fct
import sys
sys.path.append('../')
import vector_fitting
import fit_z
import fit_s


if __name__ == '__main__':
    s2p_file = 'two_coupled_SIW_ant_39GHz.s2p'
    freq, n, z, s, z0 = snp_fct.read_snp(s2p_file)
    
    