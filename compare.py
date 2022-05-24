#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri

def load_system(seedname, mode):
    if mode == "naive":
        transl_inv = False
        transl_inv_offdiag = False
    elif mode == "MV":
        transl_inv = True
        transl_inv_offdiag = False
    elif mode == "full-TI":
        transl_inv = False
        transl_inv_offdiag = True
    else:
        raise ValueError("Wrong mode")
    system = wberri.System_w90(seedname, berry=True, transl_inv=transl_inv, transl_inv_offdiag=transl_inv_offdiag, guiding_centers=True)
    return system

system_dict = {}
for shift in ["0.0", "0.5"]:
    for mode in ["naive", "MV", "full-TI"]:
        # system_dict[(shift, mode)] = load_system(f"He_unitcell_shift{shift}/He", mode)
        system_dict[(shift, mode)] = load_system(f"si_unitcell_shift{shift}/si", mode)

# system_super_dict = {}
# for mode in ["naive", "MV", "full-TI"]:
#     system_super_dict[mode] = load_system(f"He_supercell/He", mode)

np.set_printoptions(precision=8)

print("Wannier centers")
for key in system_dict.keys():
    system = system_dict[key]
    print(key, system.AA_R[0, 0, system.iR0, :])

# for key in system_super_dict.keys():
#     system = system_super_dict[key]
#     print(key, system.AA_R[1, 1, system.iR0, :])
