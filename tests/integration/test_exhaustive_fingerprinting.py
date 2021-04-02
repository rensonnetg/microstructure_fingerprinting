# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:54:10 2019

Integration testing of MF utilities.

TODO: test with noise?

@author: rensonnetg
"""
import numpy as np
import os
import sys
import time

mf_from_pkg = True
if mf_from_pkg:
    from microstructure_fingerprinting import mf_utils as mfu
else:
    path_mf = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           '..', '..', 'microstructure_fingerprinting')
    abs_path_mf = os.path.abspath(path_mf)
    if abs_path_mf not in sys.path:
        sys.path.insert(0, abs_path_mf)
    import mf_utils as mfu


np.random.seed(141414)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fixtures',
    )


# %% Boundary cases that should be handled properly!

def test_boundary_cases_1d():
    sqrt2 = np.sqrt(2.0)
    # With one variable
    A = np.array([[0],
                  [1],
                  [0]])
    Y = np.array([[1, 0, sqrt2/2,   0,   sqrt2/2],
                 [0, 0, -sqrt2/2,  2, sqrt2/2],
                 [0, 1, 0,         0,   0]])
    diclengths = np.array([1])
    w_exp = [0, 0, 0, 2, sqrt2/2]
    obj_exp = [1, 1, 1, 0, 0.5]
    w_st = np.zeros(Y.shape[1])
    obj_st = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        y = Y[:, i]
        (w_st[i], ind_subdic, ind_totdic, obj_st[i],
         y_rec) = mfu.solve_exhaustive_posweights(A, y, diclengths)
    chk_w = np.all(np.isclose(w_exp, w_st))
    chk_obj = np.all(np.isclose(obj_exp, obj_st))
    assert chk_w, "Problem with w in 1-var case"
    assert chk_obj, "Problem with obj in 1-var case"


def test_boundary_cases_2d():
    # With two variables (i.e. a1 and a2 but could be in any dimension)
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)
    A = np.array([[0.5, sqrt3*0.5], [sqrt3*0.5, 0.5]])  # (d, 2)
    Y = np.array([[-sqrt3/2, 0.5, -1, -sqrt3/2, 0.5001,
                   0.5, sqrt3/2, sqrt2/2, -sqrt2/2.0],
                  [0.5, -sqrt3/2, 0, 0.5001, -sqrt3/2,
                   sqrt3/2, 0.5, sqrt2/2, -sqrt2/2.0]])  # (d, N)
    diclengths = np.array([1, 1])
    # Expected weights objective function values:
    w_exp = np.array([[0, 0], [0, 0], [0, 0], [8.66025404e-05, 0],
                      [0, 8.66025404e-05], [1, 0], [0, 1],
                      [0.51763809, 0.51763809], [0, 0]]).transpose()
    obj_exp = np.array([1, 1, 1, 1.0001000025, 1.0001000025, 0, 0, 0, 1])
    w_st = np.zeros((2, Y.shape[1]))
    obj_st = np.zeros(Y.shape[1])  # sum of squares
    for i in range(Y.shape[1]):
        y = Y[:, i]
        (w_st[:, i],
         ind_subdic,
         ind_totdic,
         obj_st[i],
         y_rec) = mfu.solve_exhaustive_posweights(A, y, diclengths)
    chk_w = np.all(np.isclose(w_st, w_exp))
    chk_obj = np.all(np.isclose(obj_st, obj_exp))
    assert chk_w, "Problem with weights in 2-var case"
    assert chk_obj, "Problem with objective function in 2-var case"


# %% Random voxels with noise, larger matrices

def test_synthetic_data():
    Nfasc = 2
    iso_on = 1
    Natoms = 700
    N_mris = 200
    Nvox = 5

    A = np.random.randn(N_mris * (Nfasc * Natoms + iso_on))
    A = A.reshape((N_mris, Nfasc * Natoms + iso_on), order='F')

    # Prepare groundtruth solutions
    ID_gt = np.zeros((Nfasc+iso_on, Nvox), dtype=int)
    ID_gt[0, :] = np.random.randint(0, Natoms, (Nvox))
    ID_gt2 = np.random.randint(0, Natoms, (Nvox))
    if Nfasc > 1:
        ID_gt[1, :] = ID_gt2 + Natoms
    if iso_on:
        ID_gt[Nfasc, :] = Nfasc*Natoms
    w_gt = np.random.rand(Nfasc+iso_on, Nvox)  # no need to normalize
    Y = np.zeros((N_mris, Nvox))
    for i in range(Nvox):
        Y[:, i] = np.dot(A[:, ID_gt[:, i]], w_gt[:, i])
    # Add and store noise
    noise = 0.1 * (2.0*np.random.rand(N_mris, Nvox)-1.0)
    Ynoisy = Y + noise
    noise_sq_nrm = np.sum(noise**2, axis=0)  # ||y-y_noisy||_2^2 for each voxel

    # Prepare dictionary for exhaustive multi-compartment fingerprinting
    # estimation
    diclengths = np.array(np.tile(Natoms, Nfasc))
    if iso_on:
        diclengths = np.append(diclengths, 1)

    # Noiseless estimation
    ID_totdic_est = np.zeros((Nfasc+iso_on, Nvox))
    for i in range(Nvox):
        (w,
         ID_subdic,  # ind_atoms_subdic
         ID_totdic_est[:, i],  # ind_atoms_totdic,
         min_obj,
         y_reconst) = mfu.solve_exhaustive_posweights(A, Ynoisy[:, i],
                                                      diclengths)
    chk_noiseless = np.all(ID_gt == ID_totdic_est)
    msg = ('Problem exhaustive fingerprinting on noiseless synthetic data')
    assert chk_noiseless, msg

    # Noisy estimation
    min_obj = np.zeros(Nvox)  # sum of squared differences
    for i in range(Nvox):
        (w,
         ID_subdic,  # ind_atoms_subdic
         ID_totdic,  # ind_atoms_totdic,
         min_obj[i],
         y_reconst) = mfu.solve_exhaustive_posweights(A, Ynoisy[:, i],
                                                      diclengths)
    msg = ('Problem exhaustive fingerprinting with groundtruth having'
           ' 1 non-zero weight per subdictionary in the presence '
           'of noise. Exhaustive solution should do better than '
           'the physical groundtruth.')
    assert np.all(min_obj < noise_sq_nrm), msg


# %% Human Connectome Project (HCP) Massachusetts General Hospital dictionary

# Solving one instance using exhaustive combinatorial non-negative
# linear least squares with 3 sub-dictionaries (2 fascicles and an isotropic
# CSF compartment)
# Dictionary (hexagonal packing)

def test_hcp_dict():
    ld_singfasc_dic = mfu.loadmat(os.path.join(FIXTURE_DIR,
                                               'MC_dictionary_hcp.mat'))
    dic_mgh = ld_singfasc_dic['dic_fascicle_refdir']
    refdir = np.array([0.0, 0.0, 1.0])
    Natoms = dic_mgh.shape[1]
    WM_DIFF = ld_singfasc_dic['WM_DIFF']
    S0_fasc = ld_singfasc_dic['S0_fascicle']
    sig_CSF = ld_singfasc_dic['sig_csf']
    subinfo = ld_singfasc_dic['subinfo']

    sch_mat = mfu.import_PGSE_scheme(os.path.join(FIXTURE_DIR,
                                     'hcp_mgh_1003.scheme1'))
    num_B0 = 40
    sch_mat_b0 = np.vstack((np.zeros((num_B0, sch_mat.shape[1])),
                            sch_mat))
    sch_mat_b0[:num_B0, 4:] = sch_mat[0, 4:]
    num_mri = sch_mat_b0.shape[0]

    Nfasc = 2
    iso_on = 1

    # Groundtruth
    i_gt = 86
    fascdirs = np.random.randn(3, Nfasc)
    fascdirs = fascdirs/np.sqrt(np.sum(fascdirs**2, axis=0, keepdims=True))
    nu_gt = np.random.rand(Nfasc + iso_on)
    nu_gt = nu_gt/np.sum(nu_gt)
    y_gt = np.zeros(num_mri)
    M0_gt = 500
    for ifasc in range(Nfasc):
        y_gt += M0_gt * nu_gt[ifasc] * mfu.rotate_atom(dic_mgh[:, i_gt],
                                                       sch_mat_b0,
                                                       refdir,
                                                       fascdirs[:, ifasc],
                                                       WM_DIFF,
                                                       S0_fasc[:, i_gt])

    # Prepare estimation, assembling total multi-compartment dictionary
    sub_dic_sizes = [Natoms] * Nfasc
    dictionary = np.zeros((num_mri, Nfasc * Natoms + iso_on))
    for ifasc in range(Nfasc):
        st = ifasc * Natoms
        end = (ifasc+1) * Natoms
        dictionary[:, st:end] = mfu.rotate_atom(dic_mgh,
                                                sch_mat_b0,
                                                refdir,
                                                fascdirs[:, ifasc],
                                                WM_DIFF,
                                                S0_fasc)
    if iso_on:
        sub_dic_sizes.append(1)
        dictionary[:, -1] = sig_CSF
        y_gt += M0_gt * nu_gt[-1] * sig_CSF
    sub_dic_sizes = np.array(sub_dic_sizes)

    DWI = y_gt

    start = time.time()
    (w_nnz,
     ind_subdic,
     ind_totdic,
     min_obj,
     y_recons) = mfu.solve_exhaustive_posweights(dictionary,
                                                 DWI,
                                                 sub_dic_sizes)
    M0_est = np.sum(w_nnz)
    if M0_est > 0:
        nu = w_nnz/M0_est
    else:
        nu = w_nnz  # all zeros
    time_el = time.time() - start
    res_str = "\n".join("Fasc %d: nu %3.2f vs %3.2f rad %3.2e vs %3.2e "
                        "fin %3.2f vs %3.2f." %
                        (i+1, nu[i], nu_gt[i],
                         subinfo['rad'][ind_subdic[i]], subinfo['rad'][i_gt],
                         subinfo['fin'][ind_subdic[i]], subinfo['fin'][i_gt])
                        for i in range(Nfasc))
    if iso_on:
        res_str += "\nCSF: nu %3.2f vs %3.2f." % (nu[-1], nu_gt[-1])
    print("Noiseless scenario, %d atoms, sol in %g [s]:\n%s" %
          (Natoms, time_el, res_str))

    res_chk = (np.all([ind_subdic[i] == i_gt for i in range(Nfasc)]) &
               np.all(np.isclose(nu_gt, nu)))

    assert res_chk, "Bad solution:\n%s" % (res_str,)
