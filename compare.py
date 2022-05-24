#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri

folder = "/home/jmlim/position/transl_inv/si"

def load_system(folder, prefix, trinv, force_reset=False):
    try:
        with open(f'{folder}/system_{trinv}.pkl', 'rb') as f:
            system = pickle.load(f)
        if force_reset:
            raise FileNotFoundError
    except FileNotFoundError:
        system = wberri.System_w90(f'{folder}/si', berry=True, transl_inv_offdiag=trinv, guiding_centers=True)
        with open(f'{folder}/system_{trinv}.pkl', 'wb') as f:
            pickle.dump(system, f, pickle.HIGHEST_PROTOCOL)
    return system

force_reset = False

system_unshifted = load_system("unshifted", "si", False)
system_shifted = load_system("shifted", "si", False)
system_unshifted_trinv = load_system("unshifted", "si", True, force_reset)
system_shifted_trinv = load_system("shifted", "si", True, force_reset)

err = np.max(abs(system_unshifted.AA_R - system_unshifted.conj_XX_R(system_unshifted.AA_R)))
print("system_unshifted Hermitian error = ", err)
err = np.max(abs(system_shifted.AA_R - system_shifted.conj_XX_R(system_shifted.AA_R)))
print("system_shifted Hermitian error = ", err)
err = np.max(abs(system_unshifted_trinv.AA_R - system_unshifted_trinv.conj_XX_R(system_unshifted_trinv.AA_R)))
print("system_unshifted_trinv Hermitian error = ", err)
err = np.max(abs(system_shifted_trinv.AA_R - system_shifted_trinv.conj_XX_R(system_shifted_trinv.AA_R)))
print("system_shifted_trinv Hermitian error = ", err)


def get_matrix(system):
    iR0 = system.iR0
    x = system.AA_R[:, :, iR0, 0].real
    x = x - np.diag(np.diag(x))

    # iR = system.iRvec.tolist().index([-1, 0, 1])
    # x = system.AA_R[:, :, iR, 0].real
    return x


# for iR in range(system_unshifted_trinv.AA_R.shape[2]):
#     if np.linalg.norm(system_unshifted_trinv.AA_R[:, :, iR, 0]) > 1E-1:
#         print(iR, system_unshifted_trinv.iRvec[iR, :], np.linalg.norm(system_unshifted_trinv.AA_R[:, :, iR, 0]))


fig, axes = plt.subplots(2, 3, figsize=(8, 6))
vmax = 0.05

iR0 = system_unshifted.iR0
x1 = get_matrix(system_unshifted)
axes[0, 0].imshow(x1, origin="upper", vmin=-vmax, vmax=vmax, cmap="bwr")
x2 = get_matrix(system_shifted)
axes[0, 1].imshow(x2, origin="upper", vmin=-vmax, vmax=vmax, cmap="bwr")
im = axes[0, 2].imshow(abs(x2 - x1), cmap="hot")
plt.colorbar(im, ax=axes[0, 2])

x1 = get_matrix(system_unshifted_trinv)
axes[1, 0].imshow(x1, origin="upper", vmin=-vmax, vmax=vmax, cmap="bwr")
x2 = get_matrix(system_shifted_trinv)
axes[1, 1].imshow(x2, origin="upper", vmin=-vmax, vmax=vmax, cmap="bwr")
im = axes[1, 2].imshow(abs(x2 - x1), cmap="hot")
plt.colorbar(im, ax=axes[1, 2])

# plt.colorbar()
plt.show()