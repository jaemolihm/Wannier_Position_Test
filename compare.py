#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import numpy as np
import matplotlib.pyplot as plt
import wannierberri as wberri
import scipy
np.set_printoptions(precision=8)

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

from wannierberri.__w90_files import CheckPoint, MMN

def compute_stengel(seedname):
    # =======================================================================
    # BEGIN Do not change these parts unless you know what you are doing
    chk = CheckPoint(seedname)
    mmn = MMN(seedname)
    mmn.set_bk(chk)

    # Unify the b vector indices
    bk_latt = np.rint((mmn.bk_cart @ np.linalg.inv(chk.recip_lattice)) * chk.mp_grid[None, None, :]).astype(int)
    bk_latt_unique = np.unique(bk_latt.reshape(-1, 3), axis=0)
    bk_cart_unique = (bk_latt_unique / chk.mp_grid[None, :]) @ chk.recip_lattice
    assert bk_latt_unique.shape == (mmn.NNB, 3)
    bk_latt_unique = [tuple(b) for b in bk_latt_unique]
    wb = np.zeros(mmn.NNB)
    ik_kb = np.zeros((chk.num_kpts, mmn.NNB), dtype=int)

    mmat_kb = np.zeros((chk.num_kpts, chk.num_wann, chk.num_wann, mmn.NNB), dtype=complex)
    for ik in range(chk.num_kpts):
        for ib in range(mmn.NNB):
            iknb = mmn.neighbours[ik, ib]
            data = mmn.data[ik, ib]
            AAW = chk.wannier_gauge(data, ik, iknb)

            # Find unique b index
            b_latt = np.rint((mmn.bk_cart[ik, ib, :] @ np.linalg.inv(chk.recip_lattice)) * chk.mp_grid).astype(int)
            ib_unique = bk_latt_unique.index(tuple(b_latt))
            assert np.allclose(bk_cart_unique[ib_unique, :], mmn.bk_cart[ik, ib, :])
            wb[ib_unique] = mmn.wk[ik, ib]
            mmat_kb[ik, :, :, ib_unique] = AAW[:, :]
            ik_kb[ik, ib_unique] = mmn.neighbours[ik, ib]
    bk = bk_cart_unique

    nk = chk.num_kpts
    nw = chk.num_wann
    nb = mmn.NNB
    # END Do not change these parts unless you know what you are doing
    # =======================================================================

    # Now, we have the following well-known quantites.
    # nk: number of k points
    # nw: number of Wannier functions
    # nb: number of b vectors
    # mmat_kb[ik, m, n, ib] = <u_mk|u_nk+b>
    # wb[ib] = w_b
    # bk[ib, idir] = b[idir]
    # ik_kb[ik, ib]: Index of k+b in the k points

    rlatt = np.zeros(list(chk.mp_grid) + [3], dtype=int)
    rlatt[:, :, :, 0] = np.arange(chk.mp_grid[0])[:, None, None]
    rlatt[:, :, :, 1] = np.arange(chk.mp_grid[1])[None, :, None]
    rlatt[:, :, :, 2] = np.arange(chk.mp_grid[2])[None, None, :]
    rlatt = rlatt.reshape((nk, 3))

    mmat_b = np.sum(mmat_kb, axis=0) / nk

    rave = np.zeros((nw, 3))
    for ib in range(nb):
        rave += -np.log(np.diagonal(mmat_b[:, :, ib])).imag[:, None] * wb[ib] * bk[ib, None, :]

    r = np.zeros((nw, nw, 3), dtype=complex)
    for ib in range(nb):
        r += 1j * scipy.linalg.logm(mmat_b[:, :, ib])[:, :, None] * wb[ib] * bk[ib, None, None, :]

    r_new = np.zeros((nw, nw, 3), dtype=complex)
    for ib in range(nb):
        mmat_full = np.zeros((nw, nk, nw, nk), dtype=complex)
        for iR in range(nk):
            for jR in range(nk):
                for ik in range(nk):
                    ikb = ik_kb[ik, ib]
                    phase = np.exp(1j * 2 * np.pi * (np.dot(rlatt[iR], chk.kpt_latt[ik] - np.dot(rlatt[jR], chk.kpt_latt[ikb]))))
                    mmat_full[:, iR, :, jR] += mmat_kb[ik, :, :, ib] * phase
        mmat_full /= nk
        print(mmat_full[:4, 0, :4, 0])
        mmat_full = mmat_full.reshape((nw * nk, nw * nk))
        log_mmat = scipy.linalg.logm(mmat_full)
        log_mmat = log_mmat.reshape((nw, nk, nw, nk))
        log_mmat_R0 = log_mmat[:, 0, :, 0]
        print(log_mmat[:4, 0, :4, 0])

        r_new += 1j * log_mmat_R0[:, :, None] * wb[ib] * bk[ib, None, None, :]

    print(rave)
    print(np.diagonal(r).T)
    print(np.diagonal(r_new).T)


    return rave, r


# system = load_system(f"si_unitcell_shift0.0/si", "full-TI")
# system_super = load_system(f"si_supercell/si", "MV")

# rave0, r = compute_stengel("si_unitcell_shift0.0/si")
# print(rave0[0, :])
# print(r[0, 0, :])
# print(rave0)
# print(np.diagonal(r).T)

# rave, r = compute_stengel("si_unitcell_shift0.0/si")
# print(np.diagonal(r).T[:4, :])
# rave, r = compute_stengel("si_unitcell_shift1.0/si")
# print(np.diagonal(r).T[:4, :] - system.real_lattice[2, :])
# rave, r = compute_stengel("si_supercell/si")
# print(np.diagonal(r).T[:4, :])

# print(np.diagonal(system.AA_R[:, :, system.iR0, :].real).T[:4, :])
# print(np.diagonal(system_super.AA_R[:, :, system_super.iR0, :].real).T[:4, :])

rave0, r0 = compute_stengel("si_unitcell_shift0.0/si")
rave1, r1 = compute_stengel("si_unitcell_shift1.0/si")
rave1 -= system.real_lattice[2, None, :]
for iw in range(system.num_wann):
    r1[iw, iw, :] -= system.real_lattice[2, :]

print("Translational invariance for Stengel-Spaldin")
print("rave: ", np.linalg.norm(rave1 - rave0))
print("r: ", np.linalg.norm(r1 - r0))


rave0, r0 = compute_stengel("si_unitcell_shift0.0/si")
rave1, r1 = compute_stengel("si_supercell/si")

print("Size inconsistency for Stengel-Spaldin")
print(np.diagonal(r1).T[:4, :] - np.diagonal(r0).T[:4, :])
print(rave1[:4, :] - rave0[:4, :])

print(np.diagonal(r1).T[:4, :].real)
print(np.diagonal(r0).T[:4, :].real)
print(rave1[:4, :])
print(rave0[:4, :])

print("Size inconsistency for Marzari-Vanderbilt")
print(np.diagonal(system_super.AA_R[:, :, system_super.iR0, :].real).T[:4, :] - np.diagonal(system.AA_R[:, :, system.iR0, :].real).T[:4, :])

# system_dict = {}
# for shift in ["0.0", "1.0"]:
#     for mode in ["naive", "MV", "full-TI"]:
#         # system_dict[(shift, mode)] = load_system(f"He_unitcell_shift{shift}/He", mode)
#         system_dict[(shift, mode)] = load_system(f"si_unitcell_shift{shift}/si", mode)

# system_super_dict = {}
# for mode in ["naive", "MV", "full-TI"]:
#     system_super_dict[mode] = load_system(f"si_supercell/si", mode)



# print("Wannier centers")
# for key in system_dict.keys():
#     system = system_dict[key]
#     r = system.AA_R[0, 0, system.iR0, :].real
#     if key[0] == "1.0":
#         r -= system.real_lattice[2, :]
#     print(key, r)

# for key in system_super_dict.keys():
#     system = system_super_dict[key]
#     r = system.AA_R[0, 0, system.iR0, :]
#     print(key, r)
