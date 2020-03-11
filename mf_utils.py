# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:53:24 2018

Routines for microstructural estimation via microstructure fingerprinting.

SOLVERS
- non-negative (linear) least squares solutions for the 1, 2 and 3-variable
  cases almost entirely hard-coded and optimized with Numba;
- combinatorial search to solve NNLS problems subject to 1-sparsity constraints
  on sub-matrices. Optimized with Numba for the cases with 1, 2 and 3 sub-
  matrices. Low-memory implementation and calls to scipy.optimize.nnls for
  4 sub-matrices and more;
- under-determined non-negative linear least squares, i.e. with "flat" system
  matrices

MATH TOOLS
- vector manipulation scripts

DW-MRI
- rotation of single-fascicle, multi-shell ("3D") data
- rotation of single-fascicle, AxCaliber-like ("2D") data

DW-MRI VISUALIZATION
- single-fascicle, multi-shell ("3D") data
- single-fascicle, AxCaliber-like ("2D") data

I/0
- script to load Matlab-style mat files, more robust than scipy.io.loadmat

IMPORTATION:
To add this folder to the Python path and be able to load the module:

import os
import sys
path_to_mf = os.path.abspath("rel/../or/abs/path/to/MF")
if path_to_mf not in sys.path:
    sys.path.insert(0, path_to_mf)
import mf_utils as mf  # to add the content of mf_utils.py for instance

@author: rensonnetg
"""
from itertools import product
import numpy as np
import numba as nba
from scipy.interpolate import interp1d
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt


#############################################################################
# SOLVERS
#############################################################################

# Wrapper function to solve combinatorial NNLS calling numba-optimized
# routines when possible
def solve_exhaustive_posweights(A, y, dicsizes, printmsg=None):
    """Solves NNLS with 1-sparsity constraints combinatorially.

    Considering the matrix A = [A1, ..., AK], the function computes
    min_{w>=0} ||Aw - y||_2^2 subject to the 1-sparsity constraints
    (if we use w = [w1,..., wK]^T ): ||w_k||_0 = 1 for k = 1,..., K.

    In Microstructure Fingerprinting, each sub-dictionary Ak typically
    represents a single-fascicle sub-dictionary rotated along a detected
    fascicle's main orientation. Alternatively, Ak can be a single column
    modeling a CSF partial volume with isotropic diffusion. The size of each
    sub-dictionary must be provided.

    Args:
      A: 2-D NumPy array.
      y: 1-D NumPy array of length A.shape[0].
      dicsizes: 1-D NumPy array containing strictly positive integers
        representing the size of each sub-dictionary in A.
        Their sum must equal A.shape[1].
      printmsg: str to be printed at the start of the execution of the
        function. Useful on computing clusters or in parallel computing.
        Default: None.

    Returns:
      w_nneg: 1-D NumPy array containing the K non-negative weights assigned
        to the one optimal column in each sub-dictionary. To get the full
        optimal w_opt do:
            w_opt = numpy.zeros(A.shape[1])
            w_opt[ind_atoms_totdic] = w_nneg.
      ind_atoms_subdic: 1-D numy array of size K containing the index of the
        column selected (having a non-zero weight) within each sub-dictionary
        Ak, i.e. ind_atoms_subdic[k] is in [0, dicsizes[k][.
      ind_atoms_totdic: 1-D NumPy array of size K containing the indices of
        all columns with non-zero weight in A, i.e. ind_atoms_totdic[k] is in
        [0, A.shape[1][.
      min_obj: floating-point scalar equal to ||Aw_opt-y||_2^2.
      y_recons: 1-D NumPy array equal to Aw_opt, i.e. the model prediction.
    """
    # Print message (can be useful on computing clusters or in // computing)
    if printmsg is not None:
        print(printmsg, end="")

    # --- Check inputs ---
    # A should be a 2D NumPy array
    assert isinstance(A, np.ndarray), "A should be a NumPy ndarray"
    assert A.ndim == 2, "A should be a 2D array"
    # A should not have zero columns
    assert not np.any(np.all(A == 0, axis=0)), "All-zero columns detected in A"
    # A should contain floating-point numbers
    if A.dtype is not np.float64:
        A = A.astype(np.float64)
    # y should be a NumPy float64 array
    assert isinstance(y, np.ndarray), "y should be a NumPy ndarray"
    if y.dtype is not np.float64:
        y = y.astype(np.float64)
    # Refuse empty data
    assert A.size > 0 and y.size > 0, "A and y should not be empty arrays"
    # A.shape[0] should match y
    msg = ("Number of rows in A (%d) should match number of elements in y (%d)"
           % (A.shape[0], y.size))
    assert A.shape[0] == y.size, msg

    # diclengths should be a NumPy int32 array with strictly positive entries
    assert isinstance(dicsizes, np.ndarray), ("dicsizes should be a "
                                              "NumPy ndarray")
    assert np.all(dicsizes > 0), "All entries of dicsizes should be > 0"
    if dicsizes.dtype is not np.int64:
        dicsizes = dicsizes.astype(np.int64)

    # Sum of subsizes should match total size of A
    msg = ("Number of columns of A (%d) does not equal sum of size of "
           "sub-matrices in diclengths array (%d)"
           % (A.shape[1], np.sum(dicsizes)))
    assert A.shape[1] == np.sum(dicsizes), msg

    # y is often read-only when passed by multiprocessing functions such as
    # multiprocessing.Pool.starmap/map, ipyparallel.Client.map/map_async, etc.
    # This made for Numba compilation errors in lsqnonneg_1var, lsqnonneg_2var
    if y.flags['WRITEABLE'] is False:
        y = y.copy()
        y.flags.writeable = True

    # --- Call solver ---
    Nvars = dicsizes.size  # number of large-scale compartments in voxel
    if Nvars == 1:
        # Call to numba-compiled function
        (w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj,
         y_recons) = solve_exhaustive_posweights_1(A, y)
        # force arrays out
        return (np.array([w_nneg]), np.array([ind_atoms_subdic]),
                np.array([ind_atoms_totdic]), min_obj, y_recons)
    elif Nvars == 2:
        # Call to numba-compiled function, optimized for 2 variables
        return solve_exhaustive_posweights_2(A, y, dicsizes)
    elif Nvars == 3:
        # Call to numba-compiled function, optimized for 3 variables
        return solve_exhaustive_posweights_3(A, y, dicsizes)
    elif Nvars >= 4:
        # Call to built-in scipy.optimize.nnls, no numba optimization
        return solve_exhaustive_posweights_4up(A, y, dicsizes)


# Solve combinatorial nnls with 1 variable; optimized with numba.
@nba.jit(nba.types.Tuple((nba.float64,
                          nba.int32,
                          nba.int32,
                          nba.float64,
                          nba.float64[:]))
         (nba.float64[:, :], nba.float64[:]),
         nopython=True, nogil=True, cache=True)
def solve_exhaustive_posweights_1(A, y):
    """Solves combinatorial NNLS subject to 1-sparsity with 1 sub-dictionary.
    """
    # Expoiting NumPy capabilities was generally disappointing compared
    # to Numba.
    # Beats Numba for A with shape (5e3, 1e4) (5e3, 1e5) (5e4, 1e4)
    # corresponding to many acquired dMRI measurements (long columns).
    # Does slightly better when numba function decorators are removed.
    # --- start NumPy code:
#    Adoty = np.dot(y, A)  # shape (A.shape[1])
#    idx_nneg = np.where(Adoty >= 0)[0]  # shape (Nnneg)
#    A_norm_sq = np.sum(A[:, idx_nneg]**2, axis=0)  # shape (Nnneg)
#    imax = np.argmax(Adoty[idx_nneg]**2/A_norm_sq)
#    ind_atoms_subdic = idx_nneg[imax]  # shape (1)
#    ind_atoms_totdic = ind_atoms_subdic  # shape (1)
#    w_nneg = Adoty[ind_atoms_subdic]/A_norm_sq[imax]  # shape (1)
#    min_obj = np.sum(y**2) - w_nneg * Adoty[ind_atoms_subdic]  # shape (1)
#    y_recons = w_nneg * A[:, ind_atoms_totdic]  # shape (A.shape[0])
    # --- End of NumPy-only code

    # Prepare output
    w_nneg = 0.0
    ind_atoms_subdic = 0
    y_sq = np.sum(y**2)
    min_obj = y_sq

    # Solve all subproblems, just try each column of A
    for i1 in range(A.shape[1]):
        # START Solve one-variable problem:

        adoty = 0.0  # scalar
        w = 0.0
        resnorm = y_sq  # starts there, we might subtract from it
        for i in range(A.shape[0]):
            adoty = adoty + A[i, i1]*y[i]

        if adoty >= 0:
            # asq = np.dot(a, a)  # scalar
            asq = 0.0
            for i in range(A.shape[0]):
                asq = asq + A[i, i1]*A[i, i1]
            w = adoty/asq  # scalar
            resnorm -= w * adoty  # np.sum((w*a-y)**2) = y'*y - adoty^2/asq

        # END one-variable problem solved
        if resnorm < min_obj:
            ind_atoms_subdic = i1
            min_obj = resnorm
            w_nneg = w
    # Absolute index within complete dictionary A
    ind_atoms_totdic = ind_atoms_subdic
    # Reconstructed data vector
    y_recons = w_nneg*A[:, ind_atoms_totdic]
    return (w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons)


# Solve combinatorial nnls with 2 variables; optimized with Numba.
# Tries out all 2-column combinations in the double loop, no call to external
# routines.
@nba.jit(nba.types.Tuple((nba.float64[:], nba.int32[:], nba.int32[:],
                          nba.float64, nba.float64[:]))
         (nba.float64[:, :], nba.float64[:], nba.int64[:]),
         nopython=True, nogil=True, cache=True)
def solve_exhaustive_posweights_2(A, y, diclengths):
    """Solves combinatorial NNLS subject to 1-sparsity with 2 sub-dictionaries.
    """
    N1 = diclengths[0]  # must be an int for later indexing
    N2 = diclengths[1]
    st_ind = np.array([0, N1], dtype=np.int32)

    # Prepare output
    w_nneg = np.zeros(2)
    ind_atoms_subdic = np.zeros(2, dtype=np.int32)

    # Pre-compute in batch, Numpy-style
    # A11_st = np.sum(A[:, 0:N1]**2, axis=0)  # (N1,)
    # A22_st = np.sum(A[:, N1:(N1+N2)]**2, axis=0)  # (N2,)
    # A12_st = np.transpose(A[:, 0:N1]) @ A[:, N1:(N1+N2)]  # (N1, N2)
    # Adoty_st = np.transpose(y @ A)  # (N1+N2, 1)
    # y_sq = np.dot(y, y)  # non-negative scalar

    # With one column per sub-dictionary, 10-fold acceleration in C-style:
    A11_st = np.zeros(N1)
    for i1 in range(N1):
        for k in range(A.shape[0]):
            A11_st[i1] += A[k, i1]**2
    A22_st = np.zeros(N2)
    for i2 in range(N2):
        for k in range(A.shape[0]):
            A22_st[i2] += A[k, N1+i2]**2
    A12_st = np.zeros((N1, N2))
    for i1 in range(N1):
        for i2 in range(N2):
            for k in range(A.shape[0]):
                A12_st[i1, i2] += A[k, i1] * A[k, N1+i2]
    Adoty_st = np.zeros(N1+N2)
    y_sq = 0.0
    for k in range(A.shape[0]):
        y_sq += y[k]**2
        for i in range(N1+N2):
            Adoty_st[i] += y[k] * A[k, i]

    min_obj = y_sq  # objective function if w set to 0
    # Solve all N1xN2 subproblems
    for i1 in range(N1):
        for i2 in range(N2):
            # BEGING lsqnonneg_2var
            A11 = A11_st[i1]
            A12 = A12_st[i1, i2]
            A22 = A22_st[i2]
            Y1 = Adoty_st[i1]
            Y2 = Adoty_st[N1+i2]
            A21 = A12

            # Optimal solution of relaxed problem (multiplied by determinant
            # of A'*A), determinants of the Cramer formulas for w1, w2
            w1_star_Det = A22*Y1 - A12*Y2
            w2_star_Det = A11*Y2 - A21*Y1

            w = np.zeros(2, dtype=np.float64)  # initialize as float!
            resnorm = y_sq  # starting point
            # In microstructure fingerprinting, most likely case is one
            # positive weight for each fascicle
            if w1_star_Det > 0.0 and w2_star_Det > 0.0:
                # Simple case when the projection of y onto col(A) has
                # positive weights
                Det = A11*A22-A21*A12
                w[0] = w1_star_Det/Det
                w[1] = w2_star_Det/Det
                # Compute np.sum((w[0]*A[:, i1]+w[1]*A[:, N1+i2]-y)**2)
                resnorm = (resnorm + w[0]*w[0]*A11 + w[1]*w[1]*A22
                           + 2 * (w[0]*w[1]*A12 - w[0]*Y1 - w[1]*Y2))
            elif w1_star_Det >= 0.0 and w2_star_Det <= 0.0:
                # w2 is 0.
                if Y1 >= 0.0:
                    w[0] = Y1/A11
                    resnorm = resnorm - Y1 * w[0]  # ||y||_2^2 - Y1^2/A11
            elif w1_star_Det <= 0.0 and w2_star_Det >= 0.0:
                # w1 is 0.
                if Y2 >= 0.0:
                    w[1] = Y2/A22
                    resnorm = resnorm - Y2 * w[1]  # ||y||_2^2 - Y2^2/A22
            elif w1_star_Det < 0.0 and w2_star_Det < 0.0:
                # Hardly ever happens in microstructure fingerprinting.
                # Generally, Y1 and Y2 < 0 in this case, which yields w = 0.
                # Sometimes Y1 < 0 & Y2 > 0 (=> w1 > 0) OR
                # Y1 > 0 & Y2 <0 (=> w2 > 0), but only if <a1, a2> is negative.
                # NEVER do we have Y1 and Y2 > 0 as this would violate the
                # Cauchy-Schwarz inequality A11*A22 > A12*A12.
                if Y1 > 0:
                    w[0] = Y1/A11
                    resnorm -= Y1 * w[0]  # ||y||_2^2 - Y1^2/A11
                elif Y2 > 0:
                    w[1] = Y2/A22
                    resnorm -= Y2 * w[1]  # ||y||_2^2 - Y2^2/A22

            # END lsqnonneg_2var
            if resnorm < min_obj:
                ind_atoms_subdic[0] = i1
                ind_atoms_subdic[1] = i2
                min_obj = resnorm
                w_nneg = w

    # Absolute index within complete dictionary A
    ind_atoms_totdic = st_ind + ind_atoms_subdic
    # Reconstructed data vector in col(A) space
    y_recons = np.dot(A[:, ind_atoms_totdic], w_nneg)
    return (w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons)


# Solve 2-variable nnls optimized with numba, optimized with precomputed
# quantities. Meant to be called by solve_exhaustive_posweights_3 because it
# is orders of magnitude faster than scipy.optimize.nnls on a matrix with
# 2 columns.
@nba.jit(nba.types.Tuple((nba.float64[:], nba.float64))
         (nba.float64,
          nba.float64, nba.float64, nba.float64,
          nba.float64, nba.float64),
         nopython=True, nogil=True, cache=True)
def lsqnonneg_2var_opt(y_sq, A11, A12, A22, Y1, Y2):
    """Solves 2-variable non-negative linear least squares using precomputed
    scalar quantities.

    Computes min_{w>=0} ||A*w - y||_2^2 where A has shape (m, 2) and y (m,).

    Args:
      y_sq: scalar, ||y||_2^2.
      A11: scalar, ||A[:, 0]||_2^2.
      A12: scalar, A[:, 0]^TA[:, 1]^2.
      A22: scalar, ||A[:, 1]||_2^2.
      Y1: scalar, y^TA[:, 0].
      Y2: scalar, y^TA[:, A].

    Returns:
      w: 1-D NumPy array of shape (2,) with non-negative entries.
      resnorm: ||A*w-y||_2^2 at the optimal w.
    """
    A21 = A12

    # Optimal solution of relaxed problem (multiplied by determinant of A'*A)
    w1_star_Det = A22*Y1 - A12*Y2
    w2_star_Det = A11*Y2 - A21*Y1

    w = np.zeros(2, dtype=np.float64)  # initialize as float!
    resnorm = y_sq  # resnorm is that or less
    # In microstructure fingerprinting, most likely case is one positive
    # weight for each fascicle
    if w1_star_Det > 0 and w2_star_Det > 0:
        # Simple case when the projection of y onto col(A) has positive weights
        # Most likely case in microstructure fingerprinting
        Det = A11*A22-A21*A12
        w[0] = w1_star_Det/Det
        w[1] = w2_star_Det/Det
        resnorm = (resnorm + w[0]*w[0]*A11 + w[1]*w[1]*A22
                           + 2 * (w[0]*w[1]*A12 - w[0]*Y1 - w[1]*Y2))
    elif w1_star_Det >= 0 and w2_star_Det <= 0:
        # w(2) fixed to 0.
        if Y1 >= 0:
            w[0] = Y1/A11
            resnorm = resnorm - Y1 * w[0]  # np.sum((w[0]*A[:, 0]-y)**2)
    elif w1_star_Det <= 0 and w2_star_Det >= 0:
        # w(1) fixed to 0.
        if Y2 >= 0:
            w[1] = Y2/A22
            resnorm = resnorm - Y2 * w[1]  # np.sum((w[1]*A[:, 1]-y)**2)
    elif w1_star_Det < 0 and w2_star_Det < 0:
        # Y1 > 0 and Y2 > 0 never happens here. Y1 < 0 and Y2 < 0 is the most
        # usual. Y1 and Y2 have opposite signs only if A12 < 0.
        if Y1 > 0:
            w[0] = Y1/A11
            resnorm -= Y1 * w[0]  # ||y||_2^2 - Y1^2/A11
        elif Y2 > 0:
            w[1] = Y2/A22
            resnorm -= Y2 * w[1]  # ||y||_2^2 - Y2^2/A22
    return (w, resnorm)


# Tries out all 3-column combinations in the double loop, calls to 2-variable
# routines. Speed-up factor 20 compared to scipy.optimize.nnls version using
# itertools.product on an example with y of shape (552,) and
# diclengths=[821, 782, 1].
@nba.jit(nba.types.Tuple((nba.float64[:], nba.int32[:], nba.int32[:],
                          nba.float64, nba.float64[:]))
         (nba.float64[:, :], nba.float64[:], nba.int64[:]),
         nopython=True, nogil=True, cache=True)
def solve_exhaustive_posweights_3(A, y, diclengths):
    """Solves combinatorial NNLS subject to 1-sparsity with 3 sub-dictionaries.
    """
    N1 = diclengths[0]  # must be an int for later indexing
    N2 = diclengths[1]
    N3 = diclengths[2]
    s_ind = np.array([0, N1, N1+N2], dtype=np.int32)
    # e_ind = np.array([N1, N1+N2, N1+N2+N3], dtype=np.int32)

    # Tolerance for round-off errors, for comparisons with zero
    eps = 2.2204e-16
    tol = 100 * eps

    # Prepare output
    w_nneg = np.zeros(3)
    ind_atoms_subdic = np.zeros(3, dtype=np.int32)

    # Pre-compute in batch
    # Initialization Numpy mode
#    A11_st = np.sum(A[:, s_ind[0]:e_ind[0]]**2, axis=0)  # (N1,)
#    A22_st = np.sum(A[:, s_ind[1]:e_ind[1]]**2, axis=0)  # (N2,)
#    A33_st = np.sum(A[:, s_ind[2]:e_ind[2]]**2, axis=0)  # (N3,)
#    A12_st = (np.transpose(A[:, s_ind[0]:e_ind[0]])
#              @ A[:, s_ind[1]:e_ind[1]])  # (N1, N2)
#    A13_st = (np.transpose(A[:, s_ind[0]:e_ind[0]])
#              @ A[:, s_ind[2]:e_ind[2]])  # (N1, N3)
#    A23_st = (np.transpose(A[:, s_ind[1]:e_ind[1]])
#              @ A[:, s_ind[2]:e_ind[2]])  # (N2, N3)
#    Adoty_st = np.transpose(y @ A)  # (N1+N2+N3, 1)
#    y_sq = np.dot(y, y)  # non-negative scalar

    # Initialization C style. Led to an 8-fold acceleration on examples with
    # 1 column per sub-dictionary.
    A11_st = np.zeros(N1)
    A22_st = np.zeros(N2)
    A33_st = np.zeros(N3)
    A12_st = np.zeros((N1, N2))
    A13_st = np.zeros((N1, N3))
    A23_st = np.zeros((N2, N3))
    Adoty_st = np.zeros(N1+N2+N3)
    y_sq = 0.0
    for i1 in range(N1):
        for k in range(A.shape[0]):
            A11_st[i1] += A[k, s_ind[0]+i1]**2
    for i2 in range(N2):
        for k in range(A.shape[0]):
            A22_st[i2] += A[k, s_ind[1]+i2]**2
    for i3 in range(N3):
        for k in range(A.shape[0]):
            A33_st[i3] += A[k, s_ind[2]+i3]**2
    for i1 in range(N1):
        for i2 in range(N2):
            for k in range(A.shape[0]):
                A12_st[i1, i2] += A[k, s_ind[0]+i1] * A[k, s_ind[1]+i2]
    for i3 in range(N3):  # typically N3=1, slightly faster
        for i1 in range(N1):
            for k in range(A.shape[0]):
                A13_st[i1, i3] += A[k, s_ind[0]+i1] * A[k, s_ind[2]+i3]
    for i3 in range(N3):  # typically N3=1, slightly faster
        for i2 in range(N2):
            for k in range(A.shape[0]):
                A23_st[i2, i3] += A[k, s_ind[1]+i2] * A[k, s_ind[2]+i3]
    for k in range(A.shape[0]):
        y_sq += y[k]**2
        for i in range(N1+N2+N3):
            Adoty_st[i] += y[k] * A[k, i]

    min_obj = y_sq  # objective function if w set to 0

    # Solve all N1xN2xN3 subproblems
    for i3 in range(N3):  # typically N3=1, slightly faster
        A33 = A33_st[i3]
        Y3 = Adoty_st[s_ind[2] + i3]
        for i1 in range(N1):
            A11 = A11_st[i1]
            A13 = A13_st[i1, i3]
            Y1 = Adoty_st[s_ind[0] + i1]
            for i2 in range(N2):
                # BEGING lsqnonneg_3var
                A12 = A12_st[i1, i2]
                A22 = A22_st[i2]
                A23 = A23_st[i2, i3]
                Y2 = Adoty_st[s_ind[1] + i2]

                # Cramer determinant formula for x1, x2, x3
                # ! Prone to round-off errors, use tolerance
                D1 = (Y1*(A22*A33-A23*A23) - Y2*(A12*A33-A23*A13)
                      + Y3*(A12*A23-A22*A13))
                D2 = (-Y1*(A12*A33-A13*A23) + Y2*(A11*A33-A13*A13)
                      - Y3*(A11*A23-A12*A13))
                D3 = (Y1*(A12*A23-A13*A22) - Y2*(A11*A23-A12*A13)
                      + Y3*(A11*A22-A12*A12))
                if D1 >= -tol and D2 >= -tol and D3 >= -tol:
                    # All components positive or zero, most likely case
                    # in microstructure fingerprinting
                    D = (A11*(A22*A33-A23*A23)
                         - A12*(A12*A33-A23*A13)
                         + A13*(A12*A23-A22*A13))
                    w = np.array([D1, D2, D3])/D
                    res = 0.0
                    for k in range(A.shape[0]):
                        res += (w[0]*A[k, s_ind[0]+i1]
                                + w[1]*A[k, s_ind[1]+i2]
                                + w[2]*A[k, s_ind[2]+i3]-y[k])**2
                else:
                    # Try out the three possible combinations of two columns
                    # The 2-variable routine finds the optimal solution even
                    # if it only has one active column.
                    (w12,
                     res12) = lsqnonneg_2var_opt(y_sq, A11, A12, A22, Y1, Y2)
                    w = np.array([w12[0], w12[1], 0])
                    res = res12

                    (w13,
                     res13) = lsqnonneg_2var_opt(y_sq, A11, A13, A33, Y1, Y3)
                    if res13 < res:
                        w = np.array([w13[0], 0, w13[1]])
                        res = res13

                    (w23,
                     res23) = lsqnonneg_2var_opt(y_sq, A22, A23, A33, Y2, Y3)
                    if res23 < res:
                        w = np.array([0, w23[0], w23[1]])
                        res = res23

                # END lsqnonneg_3var
                if res < min_obj:
                    ind_atoms_subdic[0] = i1
                    ind_atoms_subdic[1] = i2
                    ind_atoms_subdic[2] = i3
                    min_obj = res
                    w_nneg = w

    # Absolute index within complete dictionary A
    ind_atoms_totdic = s_ind + ind_atoms_subdic
    # Reconstructed data vector in col(A) space
    y_recons = np.dot(A[:, ind_atoms_totdic], w_nneg)
    return (w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons)


# Solve combinatorial nnls with 3 or more variables, partial numba optimization
# calls scipy.optimize.nnls numpy.prod(diclengths) times
def solve_exhaustive_posweights_4up(A, y, diclengths):
    """Solves combinatorial NNLS with 4 or more sub-dictionaries, selecting
    exactly one optimal column from each sub-dictionary.
    """
    N_LSC = diclengths.size  # number of large-scale compartments in voxel
    end_ind = np.cumsum(diclengths)  # indices excluded in Python
    st_ind = np.zeros(diclengths.size, dtype=np.int64)
    st_ind[1:] = end_ind[:-1]
    Nsubprob = np.prod(diclengths)

    #  Compute all the combinations of atoms from each fascicle sub-dictionary
    # atom_indices = arrangements(diclengths)  # too much memory used
    idx_range = tuple([np.arange(diclengths[i])
                       for i in range(len(diclengths))])

    # Prepare output
    w_nneg = np.zeros(N_LSC)
    ind_atoms_subdic = np.zeros(N_LSC, dtype=np.int64)
    y_sq = np.sum(y**2)
    min_obj = y_sq
    y_recons = np.zeros(y.shape)  # reconstructed data vector

    # Solve all subproblems. Note: do not create list of all index
    # combinations because way too memory expensive. Use itertools.product.
    cnt = 0
    for idx in product(*idx_range):
        cnt += 1
        Asmall = A[:, st_ind + idx]
        w, obj_fun_sqrt = scipy.optimize.nnls(Asmall, y)
        obj_fun = obj_fun_sqrt * obj_fun_sqrt
# # Using our nnls_underdetermined script was slower than scipy.optimize.nnls
#        w, PP, _ = nnls_underdetermined(Asmall, y)
#        Adotw = Asmall[:, PP] @ w[PP]
#        obj_fun = np.sum((y - Adotw)**2)
        if obj_fun < min_obj:
            w_nneg = w
            min_obj = obj_fun
            ind_atoms_subdic = np.array(idx)
            # y_recons = Adotw
    assert Nsubprob == cnt, "Problem with number of NNLS subproblems solved"
    # absolute index within complete dictionary A
    ind_atoms_totdic = st_ind + ind_atoms_subdic
    # reconstructed data vector
    y_recons = np.dot(A[:, ind_atoms_totdic], w_nneg)
    return (w_nneg, ind_atoms_subdic, ind_atoms_totdic,
            min_obj, y_recons)


# @nba.jit(nba.types.UniTuple(nba.float64[:], 2)(
#          nba.float64[:, :], nba.float64[:]),
#          nopython=True, nogil=True, cache=True)
def nnls_underdetermined(X, y):
    '''
    Non-negative linear least-squares for underdetermined systems.

    Solves min ||Ax-y||_2^2 under the constraint x>=0.

    Written by Gaetan Rensonnet on March 7, 2019. Adapted to horizontal system
    matrices A (more columns than rows) as in microstructure fingerprinting.

    Adapted from code by Graeme O'Keefe shared with Matplotlib user list on 25
    May 2006.
    (http://matplotlib.1069221.n5.nabble.com/python-version-of-nnls-fnnls-td3921.html)
    itself inspired by Rasmus bro's FNNLS algorithm, which is nothing more than
    a computational trick for speeding up the Lawson-Hanson with vertical
    system matrices and can hardly be called an algorithm at all.

    Args:
      A: NumPy 2-D array.
      y: NumPy 1-D array.

    Returns:
      x: NumPy 1-D array containing non-negative entries.
      PP: NumPy 1-D array containing the indices of the non-zero entries of x.
      w: NumPy 1-D array equal to half the opposite of the gradient of
          ||Ax-y||_2^2, i.e. -0.5* 2*A^T(Ax-y) = A^Ty - A^TAx. If x[i]==0 then
          w[i]<=0 and if x[i]>0 then w[i]==0.
    '''
    (m, n) = X.shape
    Xty = np.dot(X.T, y)

    # Set tolerance
    eps = 2.2204e-16
    tol = 10 * eps

    # State vector: S[i] = 1 marks a positive (passive) variable,
    # S[i] = 0 marks a zero (ative) variable
    S = np.zeros(n, np.int8)
    PP = np.zeros(0)  # empty positive (passive) set

    z = np.zeros(n, np.float64)  # new iterate with better residual
    x = np.zeros(n, np.float64)  # final non-negative solution

    # Minus gradient of 0.5*||Xx-y||_2^2 (grad of obj is 2*X^T(Xx-y) )
    w = Xty  # the X.T @ X @ Xx term is zero at first

    # outer loop to put variables into set to hold positive coefficients
    while np.any(S == 0) and np.any(w[S == 0] > tol):
        # Get variable in active (zero) set with most negative gradient
        t = np.argmax(w[S == 0])
        t = np.where(S == 0)[0][t]
        # Move that variable from the active (zero) to the passive
        # (free, positive) set
        S[t] = 1

        z[t] = 0

        PP = np.where(S == 1)[0]
        XtyPP = Xty[PP]
        if PP.size == 1:
            # Scalar case, simple division
            XtXPP_s = np.sum(X[:, PP]**2)
            z[PP] = XtyPP / XtXPP_s
        else:
            # Linear least squares without constraints
            # Reduce flops by 2 due to sym ?
            XtXPP = X[:, PP].T @ X[:, PP]
            z[PP] = np.linalg.solve(XtXPP, XtyPP)
        # end

        # inner loop to remove elements from the positive set which no
        # longer belong
        while np.any(z[S == 1] <= tol):
            # Move along line from x (feasible) to z (lower objective)
            # Find critical coordinate(s) in passive (free) set turned 0 or
            # negative
            QQ = (z <= tol) & (S == 1)  # passive and non-positive
            alpha = np.min(x[QQ] / (x[QQ] - z[QQ]))
            x += alpha * (z - x)

            # More than one coordinate from passive set might have been set
            # to zero
            pass_turned_zero = (S == 1) & (np.abs(x) < tol)
            S[pass_turned_zero] = 0
            z[pass_turned_zero] = 0

            PP = np.where(S == 1)[0]
            XtyPP = Xty[PP]
            if PP.size == 1:
                XtXPP_s = np.sum(X[:, PP]**2)
                z[PP] = XtyPP / XtXPP_s
            else:
                XtXPP = X[:, PP].T @ X[:, PP]
                z[PP] = np.linalg.solve(XtXPP, XtyPP)
        # end while inner loop
        x[:] = z  # copy *values* of z into x
        w = Xty - np.dot(X.T, np.dot(X[:, PP], x[PP]))
    # end while outer loop
    return x, PP, w


#############################################################################
# MATH TOOLS
#############################################################################


def get_perp_vector(v):
    """Returns vector(s) having a zero dot product with the vector(s) in v.

    The returned N-D array `u` has the same shape as `v` and is such that
    the dot product along their first dimension is zero::

        numpy.sum(v[:, i2, ..., iN] * u[:, i2, ..., iN]) = 0
        numpy.sum(v * u, axis=0) = numpy.zeros(v.shape[1:])

    The returned vectors have unit Euclidean norm::
        numpy.sum(u**2, axis=0) = numpy.ones(v.shape[1:])

    Args:
      v: N-D NumPy array.

    Returns:
      An N-D NumPy array.
    """

    v_perp = np.zeros(v.shape)
    is_zero = v < 10 * 2.2204e-16
    num_zeros_v = np.sum(is_zero, axis=0)
    is_nonzero_vect = num_zeros_v == 0

    # If v[:, i2, ..., iN] contains one or more zero entries,
    # v_perp[:, i2, ..., iN] can just contain ones at the corresponding
    # entries to ensure a zero dot product.
    v_perp[is_zero] = 1

    # If all entries of v[:, i2, ..., iN] are non-zero, the first
    # (v.shape[0]-1) entries of v_perp[:, i2, ..., iN] are set to 1 and the
    # last entry is set to
    #   - sum(v[:-1, i2, ..., iN])/v[-1, i2, ..., iN]
    v_perp[:-1, is_nonzero_vect] = 1
    last_elements = -np.sum(
        v[:-1, is_nonzero_vect], axis=0
        )/v[-1, is_nonzero_vect]
    v_perp[-1, is_nonzero_vect] = last_elements

    # Return normalized vectors along first dimension
    norm_v_perp = np.sqrt(np.sum(v_perp**2, axis=0))
    v_perp = v_perp/norm_v_perp
    return v_perp


def rotate_vector(v, rot_axis, theta):
    """Performs rotation in 3D defined by a rotation axis and an angle.

    Args:
      v: NumPy array with shape (3,). Vector to be rotated.
      rot_axis: NumPy array with shape (3,) and unit Euclidean norm.
        Rotation axis.
      theta: floating-point number. Rotation angle in radians.

    Returns:
      NumPy array with shape (3,). Rotated vector.

    Raises:
      ValueError: if rot_axis does not have unit Euclidean norm.
    """

    # TODO: make it compatible with v of shape (3, N)
    norm_sq_axis = np.sum(rot_axis**2)
    if ~np.isclose(1, norm_sq_axis):
        raise ValueError("rotation axis should have unit norm,"
                         " detected %g" % np.sqrt(norm_sq_axis))
    costh = np.cos(theta)

    v_rot = (costh*v + np.sin(theta)*np.cross(rot_axis, v) +
             (1-costh)*(np.dot(rot_axis, v) * rot_axis))
    return v_rot


def vrrotvec2mat(rotax, theta):
    """Rotation matrix from rotation axis and angle.
    """
    if rotax.size != 3:
        raise ValueError("rotation axis should be a 3-element NumPy array")
    if ~np.isclose(np.sum(rotax**2), 1):
        raise ValueError("rotation axis should have unit norm")
    s = np.sin(theta)
    c = np.cos(theta)
    t = 1-c
    x = rotax[0]
    y = rotax[1]
    z = rotax[2]
    m = np.array([[t*x*x + c, t*x*y - s*z, t*x*z + s*y],
                  [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
                  [t*x*z - s*y, t*y*z + s*x, t*z*z + c]])
    return m


#############################################################################
# DW-MRI
#############################################################################


def DT_col_to_2Darray(DT_col, order='row'):
    """Reformats 1-D input vector (3D) into 3x3 symmetric NumPy array.

    Args:
      DT_col: (..., 6) NumPy array, where last dimension holds the data
        defining a real-valued symmetric diffusion tensor.
      order: 'row' (default) assumes values are provided in the order
        [xx xy xz yy yz zz], 'column' assumes [xx xy yy xz yz zz] and
        'diagonal' assumes [xx yy zz xy yz xz]

    Return:
      (..., 3, 3) NumPy array.
    """
    if DT_col.shape[-1] != 6:
        raise ValueError("Last dimension of input should have size 6,"
                         " detected %d." % DT_col.shape[-1])

    out = np.zeros(DT_col.shape[:-1] + (3, 3))

    if order == 'row':
        (out[..., 0, 0], out[..., 0, 1], out[..., 0, 2],
         out[..., 1, 0], out[..., 1, 1], out[..., 1, 2],
         out[..., 2, 0], out[..., 2, 1], out[..., 2, 2]
         ) = (DT_col[..., 0], DT_col[..., 1], DT_col[..., 2],
              DT_col[..., 1], DT_col[..., 3], DT_col[..., 4],
              DT_col[..., 2], DT_col[..., 4], DT_col[..., 5])
    elif order == 'column':
        (out[..., 0, 0], out[..., 0, 1], out[..., 0, 2],
         out[..., 1, 0], out[..., 1, 1], out[..., 1, 2],
         out[..., 2, 0], out[..., 2, 1], out[..., 2, 2]
         ) = (DT_col[..., 0], DT_col[..., 1], DT_col[..., 3],
              DT_col[..., 1], DT_col[..., 2], DT_col[..., 4],
              DT_col[..., 3], DT_col[..., 4], DT_col[..., 5])
    elif order == 'diagonal':
        (out[..., 0, 0], out[..., 0, 1], out[..., 0, 2],
         out[..., 1, 0], out[..., 1, 1], out[..., 1, 2],
         out[..., 2, 0], out[..., 2, 1], out[..., 2, 2]
         ) = (DT_col[..., 0], DT_col[..., 3], DT_col[..., 5],
              DT_col[..., 3], DT_col[..., 1], DT_col[..., 4],
              DT_col[..., 5], DT_col[..., 4], DT_col[..., 2])
    else:
        raise ValueError("Unknown order option \"%s\"." % order)
    return out


def get_gyromagnetic_ratio(element='H'):
    """Gyromagnetic ratio of specified element.
    """
    if element in ['hydrogen', 'H', 'proton']:
        gamma = 2*np.pi*42.577480e6
    elif element in ['carbon', 'C']:
        gamma = 2*np.pi*10.7084e6
    elif element in ['phosphorus', 'P']:
        gamma = 2*np.pi*17.235e6
    else:
        raise ValueError('Gyromagnetic ratio for nucleus of element %s'
                         'unknown.' % element)
    return gamma


def rotate_scheme_mat(sch_mat, cyldir1, cyldir2):
    """Rotates a scheme_matrix to simulate rotation of fascicle.

    sch_mat_eff = rotate_scheme_mat(sch_mat,cyldir1, cyldir2)

    Invariant:
    DWI(fascicle(dir2); sch_mat) = DWI(fascicle(dir1), sch_mat_eff)

    useful to reuse DWI(fascicle(dir1); P) results as the variations with P
    are much easier to compute afterwards; one would need to interpolate from
    DWI(fascicle(dir1); sch_mat) to DWI(fascicle(dir1); sch_mat_eff) to get
    an approximaton of DWI(fascicle(dir2), sch_mat). In Monte Carlo
    simulations, the fascicle directionis often imposed to be the z-axis, which
    we refer to as cyldir1. The right-hand-side can then be obtained and the
    left-hand-side can be interpreted as an atom or a dictionary oriented along
    another direction, cyldir2.

    Date: July 22, 2019 (see Matlab_utils/rotate_scheme_mat.m)
    Author: Gaetan Rensonnet
    """
    if cyldir1.size != 3 or cyldir2.size != 3:
        raise ValueError("cyldir1 and cyldir2 should be 3-elements NumPy"
                         " arrays.")
    if (~np.isclose(np.sum(cyldir1**2), 1) or
            ~np.isclose(np.sum(cyldir2**2), 1)):
        raise ValueError("cyldir1 and cyldir2 should have unit norm.")
    # Rotation axis to go from default z-axis to new cyldir
    rot_ax = np.cross(cyldir1, cyldir2)

    # If cyldir already parallel to z (cross product is zero), simply return
    # original sch_mat
    sch_mat_eff = sch_mat

    rot_ax_sqrd = np.sum(rot_ax**2)
    if rot_ax_sqrd > 0:
        # z-axis and cyldir not parallel (cross product is a non-zero vector)
        rot_ax = rot_ax/np.sqrt(rot_ax_sqrd)
        rot_ang = np.arccos(np.dot(cyldir1, cyldir2))
        rot_mat = vrrotvec2mat(rot_ax, -rot_ang)
        rot_gdir = sch_mat[:, :3] @ rot_mat.T
        # ! one rotated direction could be [eps, eps, sqrt(1-2eps**2)~=1]
        rot_gdir[np.abs(rot_gdir) <= np.finfo(float).eps] = 0
        rot_gdir_norm = np.sqrt(np.sum(rot_gdir**2,
                                       axis=1, keepdims=True))  # (Nseq, 1)
        nnz_g = np.squeeze(rot_gdir_norm > 0)
        rot_gdir[nnz_g, :] = (rot_gdir[nnz_g, :]/rot_gdir_norm[nnz_g, :])
        sch_mat_eff = rot_gdir  # (Nseq, 3)
        if sch_mat.shape[1] > 3:
            sch_mat_eff = np.hstack((sch_mat_eff, sch_mat[:, 3:]))
    return sch_mat_eff


def rotate_atom(sig, sch_mat, ordir, newdir, DIFF, S0, warnings=True):
    """Rotate HARDI DW-MRI signals arising from single fascicles.

    Assumes signals split per shell can be written as a function of the dot
    product between the applied gradient orientation and the main orientation
    of the fascicle of axons. Also assumes that diffusion is FREE along the
    main orientation of the fascicle.

    Args:
      sig: NumPy 1-D or 2D array of shape (Nmris,) or (Nmris, Nsub);
      sch_mat: NumPy 2-D array of shape (Nmris, 6) or (Nmris, 7);
      ordir: NumPy 1-D array of shape (3,) specifying main orientation of sig;
      newdir: NumPy 1-D array of shape (3,) specifying the new main
          orientation of the rotated signals;
      DIFF: floating-point scalar. Used to add the free diffusion data point
          (1, E_free*S0) to stabilize interpolation in interval [0, 1];
      S0: NumPy array of the same shape as sig containing the non diffusion-
          weighted signal valules. Within a substrate and a HARDI shell, all
          values should be identical. Like DIFF, just used to add the
          free-diffusion data point to stablize the interpolation.
      warnings: whether to print warning to stdout. Default: True.

    Returns:
      sig_rot: NumPy array of the same shape as sig containing the rotated
          signals(s)

    """
    # Check inputs
    assert isinstance(sig, np.ndarray), "Input sig should be a NumPy ndarray"
    assert isinstance(sch_mat, np.ndarray), ("Input sch_mat should be a "
                                             "NumPy ndarray")
    assert isinstance(ordir, np.ndarray), ("Input ordir should be a NumPy "
                                           "ndarray")
    assert isinstance(newdir, np.ndarray), ("Input newdir should be a "
                                            "NumPy ndarray")
    # Fix stupid Python way to slice 2D arrays into 1D-none arrays (x,)
    sig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig.reshape((sig.size, 1))

    if not isinstance(DIFF, np.ndarray):
        DIFF = np.array([[DIFF]])

    assert isinstance(S0, np.ndarray), "Input S0 should be a NumPy ndarray"
    if S0.ndim == 1:
        S0 = S0[:, np.newaxis]

    if sch_mat.shape[1] < 6:
        raise ValueError('sch_mat must be a N-by-6 or7 matrix')
    if sch_mat.shape[0] != sig.shape[0]:
        raise ValueError('sch_mat and sig must have the same number of rows')
    assert sig.shape == S0.shape, ("The S0 matrix should have the same size "
                                   "as the signal matrix")

    num_subs = sig.shape[1]
    gam = get_gyromagnetic_ratio('H')  # 2*np.pi*42.577480e6
    ordirnorm = np.sqrt((ordir**2).sum())
    newdirnorm = np.sqrt((newdir**2).sum())

    Gdir_norm_all = np.sqrt((sch_mat[:, 0:3]**2).sum(axis=1, keepdims=True))
    # b0 image have gradient with zero norm. Avoid true_divide warning:
    Gdir_norm_all[Gdir_norm_all == 0] = np.inf
    orcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3]/Gdir_norm_all,
                                  ordir/ordirnorm))  # Ngrad x None
    newcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3]/Gdir_norm_all,
                                   newdir/newdirnorm))  # Ngrad

#    # EDIT
#    # Sort once and for all then extract shells
#    i_orcyldotG_srt = np.argsort(orcyldotG_all)
#    GdG_un_2, i_un_2 = np.unique(sch_mat[i_orcyldotG_srt, 3:6],
#                                 return_inverse=True, axis=0)
#    # END EDIT

    # Iterate over all unique (G, Del, del) triplets
    bvals = ((gam*sch_mat[:, 3] * sch_mat[:, 5])**2
             * (sch_mat[:, 4] - sch_mat[:, 5]/3))
    sig_rot = np.zeros(sig.shape)
    GdD_un, i_un = np.unique(sch_mat[:, 3:6], return_inverse=True, axis=0)
    num_shells = GdD_un.shape[0]

    for i in range(num_shells):
        ind_sh = np.where(i_un == i)[0]  # returns a tuple
        bval = bvals[ind_sh[0]]  # Ngrad_shell

#        # EDIT
#        ind_sh_chk = i_orcyldotG_srt[i_un_2==i]
#        assert (ind_sh_chk.shape[0]==
#                ind_sh.shape[0]), ("Problem with new shell indices "
#                                   "for shell %d/%d"%(i, num_shells))
#        # END EDIT

        # No rotation for b0 images
        if bval == 0:
            sig_rot[ind_sh, :] = sig[ind_sh, :]
            continue

        # A shell should contain at least two data points
        if ind_sh.size < 2:
            raise ValueError("Fewer than 2 identical (G, Del, del) triplets "
                             "detected for triplet %d/%d (%g, %g, %g), b=%g"
                             " s/mm^2, probably not a HARDI shell." %
                             (i+1, num_shells, GdD_un[i, 0], GdD_un[i, 1],
                              GdD_un[i, 2], bval/1e6))
        # Print warning if very few points detected in one shell
        if ind_sh.size < 10 and warnings:
            print("WARNING: rotate_atom: fewer than 10 data points detected"
                  " for acquisition parameters (G, Del, del) %d/%d "
                  "(%g, %g, %g), b=%g s/mm^2.\n"
                  "Quality of approximation may be poor."
                  % (i+1, num_shells, GdD_un[i, 0], GdD_un[i, 1],
                     GdD_un[i, 2], bval/1e6))

        # Check that non diffusion weighted values are identical in a shell
        # for each substrate separately
        S0_sh_ok = np.all(np.isclose(S0[ind_sh, :],
                                     S0[ind_sh[0], :]),
                          axis=0)  # Nsubs
        if np.any(~S0_sh_ok):
            bad_subs = np.where(~S0_sh_ok)[0]
            raise ValueError('Distinct values in provided S0 image '
                             'for shell  %d/%d (b=%g s/mm^2) '
                             'for %d substrate(s) [%s]' %
                             (i+1, num_shells, bval/1e6,
                              bad_subs.shape[0],
                              " ".join("{:d}".format(b) for b in bad_subs)))

        # Double check unicity of (G,Del,del) triplets (redundant)
        Gb = np.unique(sch_mat[ind_sh, 3])
        Delb = np.unique(sch_mat[ind_sh, 4])
        delb = np.unique(sch_mat[ind_sh, 5])
        if Gb.size > 1:
            raise ValueError('Distinct G values detected (G1=%g, G2=%g, ...) '
                             'for triplet %d/%d, b-value %g s/mm^2' %
                             (Gb[0], Gb[1], i+1, num_shells, bval/1e6))
        if Delb.size > 1:
            raise ValueError('Distinct Del values detected (D1=%g, D2=%g, ...)'
                             ' for triplet %d/%d, b=%g s/mm^2' %
                             (Delb[0], Delb[1], i+1, num_shells, bval/1e6))
        if delb.size > 1:
            raise ValueError('Distinct del values detected (d1=%g, d2=%g, ...)'
                             'for triplet %d/%d, b=%g s/mm^2' %
                             (delb[0], delb[1], i+1, num_shells, bval/1e6))

        # Sort shell data as a function of dot product with original cyl dir
        # FIXME: keep average signal for identical dot products instead of just
        # keeping one data point via use of np.unique
        (sorted_orcyl_uni,
         sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                           return_index=True)
        dot_prod_data = sorted_orcyl_uni  # Ngrad_shell x None

        newcyldotG = newcyldotG_all[ind_sh]  # Ngrad_shell x None

        sig_or_shell = sig[ind_sh, :]  # Ngrad_shell x Nsubs
        sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # Ngrad_shell x Nsubs

#        # EDIT
#        # Problem: it still does not solve the problem of duplicate x values
#        # which will cause interp1 to throw an error so some sort of call to
#        # unique would still be required...
#        # Plus it's not clear whether sorting everything at the beginning
#        # really is faster than sorting smaller chunks in this loop
#        dot_prod_data_chk = orcyldotG_all[ind_sh_chk]
#        sig_data_chk = sig[ind_sh_chk, :]
#        newcyldotG_chk = newcyldotG_all[ind_sh_chk]
#        # for newcyldotG the order would be different but that does not matter
#        # as long as the receiver order in sig_rot[ind_sh_chk]
#        # matches newcyldotG_all[ind_sh_chk]
#        assert (np.all(dot_prod_data ==
#                       dot_prod_data_chk)), "Problem with dot_prod_data"
#        assert (np.all(sig_data ==
#                       sig_data_chk)), ("Problem with sig_data for "
#                                        "shell %d/%d" % (i, num_shells))
#
#        # END EDIT

        # Add the data point (1, E_free*S0_b) to better span the interpolation
        # interval :
        if not np.any(dot_prod_data == 1):
            dot_prod_data = np.append(dot_prod_data, [1])  # Ngrad_sh+1 x 1
            if num_subs == 1:
                sig_data = np.append(sig_data,
                                     np.exp(-bval * DIFF) * S0[ind_sh[0]])
                sig_data = sig_data[:, np.newaxis]
            else:
                # DIFF is 1x1 ndarray so np.exp() is too and free diff is 2D
                free_diff = np.exp(-bval * DIFF) * S0[ind_sh[0], :]
                sig_data = np.append(sig_data,  # Ngrad_sh+1 x Nsubs
                                     free_diff, axis=0)

        # Smooth out data near dotproduct ~= 0, to avoid extrapolation
        # unstabilities due to MC variability with closely-spaced data :
        almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
        cluster_size = np.sum(almost_perp)

        # Subsitute the x values close to left-most edge by their center of
        # mass. The if statement avoids taking the mean of an empty array,
        # which returns nan in NumPy (!). Do the same with the measurements.
        if cluster_size > 1:
            dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                      dot_prod_data[cluster_size:])
            # has shape (Ngrad_sh-Nzeros) x None
            sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                         axis=0,
                                         keepdims=True),
                                 sig_data[cluster_size:, :],
                                 axis=0)

        # Check consistency of interpolation data
        if (dot_prod_data.size != sig_data.shape[0]) and warnings:
            print("WARNING: rotate_atom: problem with shapes and/or sizes"
                  " before 1D interpolation at shell %d/%d "
                  "(G=%g Del=%g del=%g)" %
                  (i+1, num_shells,
                   GdD_un[i, 0], GdD_un[i, 1], GdD_un[i, 2]))

        # Apply rotation to each substrate for the current shell in batch mode
        f_interp_shell = interp1d(dot_prod_data, sig_data, axis=0,
                                  kind='linear', fill_value='extrapolate',
                                  assume_sorted=True)
        sig_rot[ind_sh, :] = f_interp_shell(newcyldotG)

        if np.any(np.isnan(sig_rot[ind_sh, :])):
            sub_has_nan = np.any(np.isnan(sig_rot[ind_sh, :]), axis=0)
            bad_subs = np.where(sub_has_nan)[0]
            raise ValueError('Nan detected after rotation of substrate(s) '
                             'for sequence(s) {%d...%d} (bval=%g s/mm^2) '
                             'for %d substrate(s): [%s]' %
                             (ind_sh[0], ind_sh[-1], bval/1e6,
                              bad_subs.shape[0],
                              " ".join("%d" % b for b in bad_subs)))
    return np.reshape(sig_rot, sig_shape)


def rotate_atom_2Dprotocol(sig, sch_mat, refdir, newdir, DIFF):
    """Rotates signals acquired with a 2D AxCaliber-like protocol.

    What matters is the signal perpendicular to the fascicle.
    Remember that
     S(fasc(dir2), sch_mat_ez) = S(fasc(e_z), sch_mat_eff)
    for sch_mat_eff obtained by rotate_scheme_matrix.
    In our experiments, Monte Carlo simulations assume dir = e_z. Our fascicles
    were along DTI_dir and therefore
        S(fasc(DTI_dir), sch_mat) = S(fasc(ez), sch_mat_rot)

    There are therefore two ways to compute the parallel component of the
    applied gradients:
        A. Gpar = |sch_mat.gdir * DTI_dir| * sch_mat.G
        B. Gpar = |sch_mat_rot.gdir * [0;0;1]| * sch_mat_rot.G
    Numerically, method B introduces fewer round-off errors since
    sch_mat_rot was the matrix used to generate the signal.

    To compute the perpendicular component of the applied gradients:
        A.  gperp = sch_mat.gdir - (sch_mat.gdir * DTI_dir)* DTI_dir
            Gperp = ||gperp|| * sch_mat.G
        B.  gperp = sch_mat_eff.gdir - (sch_mat_eff.gz) * [0;0;1]
                     = sch_mat_eff.gxy !
            Gperp = ||gperp|| * sch_mat_eff.G
    Method B has the advantage that the perpendicular gradients are easier to
    access: they are the first two columns of the scheme matrix. Given the way
    the signals were simulated, Method B will again yield fewer numerical
    errors.

    Args:
      sig: 1-D or 2-D NumPy array. Reference signal(s) for a fascicle along
        refdir (e.g., the dictionary).
      sch_mat: 2-D NumPy array. Scheme matrix in laboratory frame.
      refdir: 3-element 1-D NumPy array with unit Euclidean norm.
      newdir: 3-element 1-D NumPy array with unit Euclidean norm. Direction of
        the fascicle for which we want to interpolate the signal.
      DIFF: diffusivity relevant for free diffusion parallel to fascicles, in
        m^2/s (e.g., for water at body temperature: 3.0e-9).
    """
    sig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    if np.any(sch_mat[:, 2] != 0):
        raise ValueError("Use the original schemefile with zeros for gz.\n"
                         "Specify the reference and new orientations "
                         "separately.")
    if sig_shape[0] != sch_mat.shape[0]:
        raise ValueError("Signal and scheme matrix must have the same "
                         "number of elements (sequences) along their first"
                         " dimension. Detected %d and %d." %
                         (sig_shape[0], sch_mat.shape[0]))
    # Constants
    zdir = np.array([0, 0, 1])
    gam = get_gyromagnetic_ratio('H')
    # Protocol values which won't change, in lab frame
    G = sch_mat[:, 3]
    Delta = sch_mat[:, 4]
    delta = sch_mat[:, 5]
    is_b0 = (G == 0)
    is_b = (G != 0)

    # Compute perpendicular and parallel components of gradients for the
    # reference fascicle (in that fascicle's space)
    sm_eff_ref = rotate_scheme_mat(sch_mat, zdir, refdir)
    g_perp_ref = sm_eff_ref[:, 0:2]  # (Nseq, 2)
    g_perp_ref_norm = np.sqrt(np.sum(g_perp_ref**2, axis=1))
    nnz_gref = g_perp_ref_norm > 0
    g_perp_ref[nnz_gref, :] = (sm_eff_ref[nnz_gref, 0:2] /
                               g_perp_ref_norm[nnz_gref][:, np.newaxis])
    G_perp_ref = G * g_perp_ref_norm
    G_par_ref = np.abs(sm_eff_ref[:, 2]) * G
    chk_ref = np.isclose(G**2, G_perp_ref**2 + G_par_ref**2)
    assert np.all(chk_ref), ("Inconsistency in parallel and perpendicular"
                             " gradient components for reference fasicle.")
    # Use that to compute perpendicular and parallel signal contributions
    b_par_ref = (gam * delta * G_par_ref)**2 * (Delta - delta/3)  # (Nseq)
    S_par_ref = np.exp(-b_par_ref * DIFF)  # (Nseq)
    S_perp_ref = sig / S_par_ref[:, np.newaxis]
    chk_par_ref = np.isclose(S_par_ref[is_b0], 1)
    assert np.all(chk_par_ref), ("Reference fascicle: parallel signal should "
                                 " be one in b0 sequences.")
    # TODO: verify against sig_par and sig_perp provided by user. Watch out for
    # shapes (sig_perp and sig_par should probably be made to have the shape
    # of sig)

    # Compute perpendicular and parallel components of gradients for the new
    # fascicle
    sm_eff_new = rotate_scheme_mat(sch_mat, zdir, newdir)
    g_perp_new = sm_eff_new[:, 0:2]  # (Nseq, 2)
    g_perp_new_norm = np.sqrt(np.sum(g_perp_new**2, axis=1))  # (Nseq,)
    nnz_gnew = g_perp_new_norm > 0
    g_perp_new[nnz_gnew, :] = (sm_eff_new[nnz_gnew, 0:2] /
                               g_perp_new_norm[nnz_gnew][:, np.newaxis])
    G_perp_new = G * g_perp_new_norm
    G_par_new = np.abs(sm_eff_new[:, 2]) * G
    chk_new = np.isclose(G**2, G_perp_new**2 + G_par_new**2)
    assert np.all(chk_new), ("Inconsistency in parallel and perpendicular "
                             "gradient components for new fascicle.")
    # Use that to compute perpendicular and parallel signal contributions
    b_par_new = (gam * delta * G_par_new)**2 * (Delta - delta/3)
    S_par_new = np.exp(-b_par_new * DIFF)
    S_par_new = S_par_new[:, np.newaxis] * np.ones((1, sig.shape[1]))
    S_perp_new = np.zeros(sig.shape)  # to be interpolated !
    # Non- DW sequences need no interpolation:
    S_perp_new[is_b0, :] = sig[is_b0, :]
    chk_par_new = np.isclose(S_par_new[is_b0, :], 1)
    assert np.all(chk_par_new), ("New fascicle: parallel signal should "
                                 " be equal to 1 in b0 sequences.")

    # Extract unique delta pairs
    Deldel_un, i_un = np.unique(sch_mat[:, 4:6], return_inverse=True, axis=0)
    num_Deldels = Deldel_un.shape[0]

    # Unique gradient directions (assumed to be in pairs of opposite
    # polarities, referred to as a "line") done for each shell in case
    # different shells use different lines. All the gdir's hereafter refer
    # to perpendicular components!
    for idel in range(num_Deldels):
        is_shell = i_un == idel
        ind_del = np.where(i_un == idel)[0]

        # Unique directions in plane perpendicular to reference fascicle
        gdir_ref_un, ig_ref_un = np.unique(g_perp_ref[ind_del, :],
                                           return_inverse=True,
                                           axis=0)  # (Nun, 2)
        # Two possibilities: 3 or 5
        # 1) ref fasc in bissector plane of the two lines:
        #        only two (opposite) perp directions + zero => 3 unique dirs
        # 2) ref fasc aligned with one of the two lines in xy plane:
        #       two (opposite) perp directions + zero => 3 unique dirs
        # 3) else: 2 pairs of opposite directions + zero => 5 unique dirs
        err_msg = ("Problem at delta pair %d/%d: found %d "
                   "unique gradient directions in plane perpendicular"
                   " to reference fascicle (including b0 zero dirs)." %
                   (idel+1, num_Deldels,
                    gdir_ref_un.shape[0]))
        assert ((gdir_ref_un.shape[0] == 5) or
                (gdir_ref_un.shape[0] == 3)), err_msg
        # Cross dot products will have a diagonal of 1s, some off-diagonal
        # elements equal to -1 (if two opposite directions are present) and
        # some others close to zero. This allows us to find pairs of opposite
        # directions by finding the -1 values.
        cross_dp = gdir_ref_un @ gdir_ref_un.T
        ig, ig_op = np.where(np.isclose(cross_dp, -1))
        assert (ig.size == 4
                or ig.size == 2), ("Problem at delta pair %d/%d: found %d "
                                   "instead of 4 (2x2, redundant) pairs of "
                                   "opposite directions in plane perpendicular"
                                   " to reference fascicle." %
                                   (idel+1, num_Deldels, ig.size))
        # ig now contains the reference unique directions to interpolate from

        # Unique directions in plane perpendicular to new fascicle
        gdir_new_un, ig_new_un = np.unique(g_perp_new[ind_del, :],
                                           return_inverse=True,
                                           axis=0)
        assert (gdir_new_un.shape[0] == 3 or
                gdir_new_un.shape[0] == 5
                ), ("Problem at delta pair %d/%d: found %d "
                    "unique gradient directions in plane perpendicular to "
                    "new fascicle (including b0 zero dirs)." %
                    (idel+1, num_Deldels,
                     gdir_new_un.shape[0]))
        # identify opposite pairs
        cross_dp_new = gdir_new_un @ gdir_new_un.T
        ipairs_new, ig_op_new = np.where(np.isclose(cross_dp_new, -1))
        is_upper_eff = ipairs_new < ig_op_new
        ipairs_new = ipairs_new[is_upper_eff]
        ig_op_new = ig_op_new[is_upper_eff]
        assert (ipairs_new.size == 2 or
                ipairs_new.size == 1), ("Problem at delta pair %d/%d: found "
                                        "%d instead of 2 pairs of opposite "
                                        "directions, in plane "
                                        " perpendicular to new fascicle." %
                                        (idel+1, num_Deldels,
                                         ipairs_new.size))
        # ipairs_new now contains the unique directions to interpolate to

        # Gradients with zero perpendicular component: take average of b0
        # signals in that shell (excluding the purely b0 measurements)
        is_vanished_new = ~nnz_gnew & is_b & is_shell
        is_sh_b0_ref = is_b0 & is_shell
        if np.sum(is_vanished_new) > 0:
            err_msg = ("Shell %d/%d: some new line directions are completely"
                       " parallel to new fascicle, implying free diffusion. "
                       "However, no b0 measurements in the reference signal"
                       " are available for this shell. We therefore can't"
                       " properly scale the new signal." %
                       (idel+1, num_Deldels))
            assert np.sum(is_sh_b0_ref) > 0, err_msg
            if np.sum(is_sh_b0_ref) == 1:
                # no need for the average. Avoid dimension problems
                S_perp_new[is_vanished_new, :] = sig[is_sh_b0_ref, :]
            else:
                mean_b0_ref = np.mean(sig[is_sh_b0_ref, :],
                                      axis=0)  # (Natoms,)
                S_perp_new[is_vanished_new, :] = mean_b0_ref

        # Loop over new line directions in perpendicular plane
        for i_line_new in range(ipairs_new.size):
            linedir_new = gdir_new_un[ipairs_new[i_line_new], :]
            # Select gradients in protocol that are
            # - in the right Delta-delta shell
            # - aligned with the line direction along which the signal must
            #   be interpolated (i.e. directions equal to or opposite the line)
            ind_new = ind_del[(ig_new_un == ipairs_new[i_line_new])
                              | (ig_new_un == ig_op_new[i_line_new])]
            assert np.all(is_b[ind_new]), ("Problem at delta pair %d/%d, "
                                           "new line direction %d/%d:"
                                           " trying to interpolate "
                                           "b0 sequences." %
                                           (idel+1, num_Deldels,
                                            i_line_new, ipairs_new.size))
            G_signed_new = (G_perp_new[ind_new] *
                            np.sign(g_perp_new[ind_new, :] @ linedir_new))
            # Find direction in plane perpendicular to reference fascicle
            # closest to the new line direction we are sampling
            i_max = np.argmax(gdir_ref_un @ linedir_new)
            line_ref = gdir_ref_un[i_max, :]
            idirref = np.where(i_max == ig)[0]  # (1) for finding the opposite
            ind_ref = ind_del[(ig_ref_un == ig[idirref]) |
                              (ig_ref_un == ig_op[idirref])]
            G_signed_ref = (G_perp_ref[ind_ref] *
                            np.sign(g_perp_ref[ind_ref, :] @ line_ref))
            # DEBUG information:
#            print("Shell %d/%d - line %d/%d" %
#                  (idel, num_Deldels, i_line_new, ipairs_new.size))
#            print("Reference sequence indices:")
#            print(" ".join("%d" % index for index in ind_ref))
#            print("New sequence indices:")
#            print(" ".join("%d" % index for index in ind_new))
#            print("Reference signed gradient intensity:")
#            print(" ".join("%4.3e" % e for e in G_signed_ref))
#            print("New signed gradient intensity:")
#            print(" ".join("%4.3e" % e for e in G_signed_new))

            # Apply interpolation to each substrate for the current shell
            # in batches
            f_interp_line = interp1d(G_signed_ref,
                                     S_perp_ref[ind_ref, :],
                                     axis=0,
                                     kind='linear',
                                     fill_value='extrapolate',
                                     assume_sorted=False)
            S_perp_new[ind_new, :] = f_interp_line(G_signed_new)
        # end of loop on lines
    # end of loop on delta-Delta shells
    S_par_new = np.reshape(S_par_new, sig_shape)
    S_perp_new = np.reshape(S_perp_new, sig_shape)
    sig_new = S_par_new * S_perp_new
    return sig_new  # , S_perp_new, S_par_new


def interp_PGSE_from_multishell(sch_mat, newdir,
                                sig_ms=None, sch_mat_ms=None, ordir=None,
                                msinterp=None):
    """Single-fascicle PGSE signal interpolated from dense multi-HARDI.

    This script can also be used to perform rotations based on the dense
    sampling rather than on the dictionary along the canonical direction.

    Args:
      sch_mat: 2D NumPy array with shape (Nseq, 7)
      newdir: NumPy array with 3 elements having unit Euclidean norm.
      sig_ms: 2D NumPy array with shape (Nsampling, Nsubs) or a 1D array with
        shape (Nsampling,).
      sch_mat_ms: 2D NumPy array with shape (Nsampling, 7)
      ordir: NumPy array with 3 elements having unit Euclidean norm.
      msinterp: interpolator object such as returned by
        init_PGSE_multishell_interp. If set, sig_ms, sch_mat_ms and ordir need
        not be set (they will be ignored if provided) and only sch_mat and
        newdir are required.

    Returns:
      A 2D NumPy array with shape (Nseq, Nsub) passed to NumPy.squeeze(),
      i.e., the resulting array could be 1D (Nseq,) or even a scalar if Nseq
      and Nsub are 1.

    """
    FAST_MODE = msinterp is not None

    if FAST_MODE:  # initialized mode
        if (msinterp['Gms_un'].size !=
                len(msinterp['interpolators'])):
            raise ValueError("msinterp['Gms_un'] has size %d vs "
                             "expected %d to match "
                             "len(msinterp['shell_interpolators'])"
                             % (msinterp['Gms_un'].size,
                                len(msinterp['shell_interpolators'])))
        # Check interpolator object
        num_subs_chk = msinterp['interpolators'][0].y.shape[1]
        if num_subs_chk != msinterp['num_subs']:
            raise ValueError("Inconsistency in msinterp "
                             "regarding number of substrates. "
                             "Make sure the interpolator was initialized"
                             " on the right dictionary.")
        num_subs = msinterp['num_subs']
        sch_DeldelTE = msinterp['scheme_DeldelTE']
    else:  # non-initialized mode
        msinterp = {}  # must be initialized
        if sig_ms is None or sch_mat_ms is None or ordir is None:
            raise ValueError("If msinterp is not specified, sig_ms, "
                             "sch_mat_ms and ordir must all be specified.")
        # Check gradient norm of dense multishell scheme
        Gdir_norm_ms = np.sqrt(np.sum(sch_mat_ms[:, 0:3]**2, axis=1))
        if np.any(np.abs(1 - Gdir_norm_ms[Gdir_norm_ms > 0]) > 1e-3):
            raise ValueError("Gradient directions in multi-shell scheme"
                             " matrix should all either have zero "
                             "or unit norm.")
        if ordir.size != 3:
            raise ValueError("Direction of dictionary computed with dense"
                             " sampling (ordir) should have 3 entries.")
        if ordir.ndim > 1:
            # Get rid of singleton dimensions because calls to 1D
            # interpolators below will be affected and dimension
            # mismatch will occur
            ordir = np.squeeze(ordir)
        # Check consistency of the timing parameters in the acquisitions
        sch_DeldelTE = sch_mat_ms[0, 4:7]
        chk_ms = np.isclose(sch_DeldelTE, sch_mat_ms[:, 4:7])
        if not np.all(chk_ms):
            raise ValueError("Delta, delta and TE values should all be "
                             "identical in multi-shell sampling.")
        # Get inputs in right format
        if sig_ms.ndim == 1:
            sig_ms = sig_ms.reshape((sig_ms.size, 1))
        if sch_mat_ms.shape[0] != sig_ms.shape[0]:
            raise ValueError("Number of lines in dense multishell scheme"
                             " (%d) does not match number of signal values"
                             " per substrate (%d)." %
                             (sch_mat_ms.shape[0], sig_ms.shape[0]))
        ordirnorm = np.sqrt((ordir**2).sum())  # (1,)
        if np.abs(1 - ordirnorm) > 1e-3:
            raise ValueError("Orientation vector of the multi-shell signal"
                             " must have unit norm. Detected"
                             " %g." % (ordirnorm,))
        num_subs = sig_ms.shape[1]
        # Dot-products between fascicle directions and gradient directions
        orcyldotG_all = np.abs(np.dot(sch_mat_ms[:, 0:3], ordir))  # (Nmsh,)
        # Find unique G values in dense sampling
        msinterp['Gms_un'], i_Gms = np.unique(sch_mat_ms[:, 3],
                                              return_inverse=True)
    # At this point, we have initialized
    #  num_subs, sch_DeldelTE, msinterp['Gms_un'], i_Gms, ordir, newdir

    # W/ or W/o initialization: check inputs
    chk_new = np.isclose(sch_DeldelTE, sch_mat[:, 4:7])
    if not np.all(chk_new):
        raise ValueError("Delta, delta and TE values should all be "
                         "identical to those in the multi-shell sampling.")
    if newdir.size != 3:
        raise ValueError("Direction of fascicle for new signal (newdir)"
                         " should have 3 entries.")
    if newdir.ndim > 1:
        # Get rid of singleton dimensions because calls to 1D
        # interpolators below will be affected and dimension
        # mismatch will occur
        newdir = np.squeeze(newdir)
    newdirnorm = np.sqrt((newdir**2).sum())  # (1,)
    if np.abs(1 - newdirnorm) > 1e-3:
        raise ValueError("Orientation vector "
                         "of the new signal must have unit norm. Detected"
                         " %g." % (newdirnorm,))
    # Check gradient norms of new acquisition scheme
    Gdir_norm_new = np.sqrt(np.sum(sch_mat[:, 0:3]**2, axis=1))
    if np.any(np.abs(1 - Gdir_norm_new[Gdir_norm_new > 0]) > 1e-3):
        raise ValueError("Gradient directions in multi-shell scheme matrix"
                         " should all either have zero or unit norm.")

    # Dot-products between new fascicle and new gradient directions
    newcyldotG_all = np.abs(np.dot(sch_mat[:, 0:3], newdir))  # (Nmri,)
    # Find unique G values in new scheme
    G_un, i_G = np.unique(sch_mat[:, 3], return_inverse=True)

    # Establish mapping: which G in multi-shell samping to use for
    # each new G?
    Gnew_to_Gms = []
    Gms_used = np.full(msinterp['Gms_un'].shape,
                       False,
                       dtype=bool)  # just to speed up non-initialized mode
    for Gnew in G_un:
        # Check for precisely same value in multi-shell sampling
        i = np.where(Gnew == msinterp['Gms_un'])[0]
        if i.size > 0:
            Gnew_to_Gms.append(i)  # i is np.array of shape (1,)
            Gms_used[i] = True
        else:
            # No identical G found, find surrounding values in mshell
            # sampling
            i_high = np.argmax(msinterp['Gms_un'] > Gnew)
            if i_high == 0:
                raise ValueError("Gradient intensity %g is not in the [%g, %g]"
                                 " range spanned by the multi-shell sampling."
                                 " Extrapolation not supported." %
                                 (Gnew,
                                  msinterp['Gms_un'][0],
                                  msinterp['Gms_un'][-1]))
            Gnew_to_Gms.append([i_high - 1, i_high])
            Gms_used[i_high - 1] = True
            Gms_used[i_high] = True

    if not FAST_MODE:
        # Compute spherical interpolator for each shell of the dense sampling
        # which will be used for the new signals
        msinterp['interpolators'] = []
        for i in range(msinterp['Gms_un'].shape[0]):
            if not Gms_used[i]:
                # Skip if it's not even going to be used in the interpolation
                msinterp['interpolators'].append(None)
                continue

            # Get lines of the multi-shell dense sampling
            ind_sh = np.where(i_Gms == i)[0]  # (Ngr_sh,)

            # If G==0, check that all G=0 signals identical in each column
            # and do not compute a spherical interpolator because it does
            # not make sense
            if msinterp['Gms_un'][i] == 0:
                chk = np.all(np.isclose(sig_ms[ind_sh, :],
                                        sig_ms[ind_sh[0], :]),
                             axis=0)  # (Nsubs,)
                if np.any(~chk):
                    bad_subs = np.where(~chk)[0]
                    raise ValueError('Distinct signal values in provided '
                                     'multi-shell sampling for zero gradients,'
                                     ' for %d substrate(s) [%s]' %
                                     (bad_subs.shape[0],
                                      " ".join("{:d}".format(b)
                                               for b in bad_subs)))
                # For b=0 (G=0) shells, an interpolator is just a constant
                # function. We just repeat, for each substrate, the first
                # signal value twice (in case only one b=0 acquisition present
                # in dense multi-shell sampling, which is likely for any
                # efficient sampling) for the x-axis points 0 and 1, the
                # extreme values of the dot product
                f_interp_shell = interp1d([0, 1],  # data below is (2, Nsubs)
                                          np.repeat([sig_ms[ind_sh[0], :]],
                                                    2, axis=0),
                                          axis=0,
                                          kind='linear',
                                          fill_value='extrapolate',
                                          assume_sorted=True)
                msinterp['interpolators'].append(f_interp_shell)
                continue
            # General G > 0 case in dense multi-shell sampling
            (sorted_orcyl_uni,
             sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                               return_index=True)
            dot_prod_data = sorted_orcyl_uni  # (Ngr_sh,)
            sig_or_shell = sig_ms[ind_sh, :]  # (Ngr_sh, Nsubs)
            sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # (Ngr_sh, Nsubs)

            # TODO: Add the data point (1, E_free*S0_b)

            # Smooth out data near dotproduct ~= 0, to avoid extrapolation
            # unstabilities due to MC variability with closely-spaced data :
            almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
            cluster_size = np.sum(almost_perp)

            # Substitute the x values close to left-most edge by their
            # center of mass. The if statement avoids taking the mean of an
            # empty array, which returns nan in Numpy (!). Do the same with
            # the measurements.
            if cluster_size > 1:
                dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                          dot_prod_data[cluster_size:])
                sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                             axis=0,
                                             keepdims=True),
                                     sig_data[cluster_size:, :],
                                     axis=0)  # ((Ngr_sh-Nzeros), Nsubs)
            # Compute spherical interpolator of shell
            f_interp_shell = interp1d(dot_prod_data,
                                      sig_data,
                                      axis=0,
                                      kind='linear',
                                      fill_value='extrapolate',
                                      assume_sorted=True)
            msinterp['interpolators'].append(f_interp_shell)
    # END of if not FAST_MODE

    # Perform interpolation inside thick spherical shells
    # First on the shells of the multi-shell sampling, then along G between
    # shells
    sig_new = np.zeros((sch_mat.shape[0], num_subs))
    for i in range(G_un.shape[0]):
        # Lines of new protocol
        ind_sh = np.where(i_G == i)[0]
        newcyldotG = newcyldotG_all[ind_sh]  # (Ngrad_shell,)

        # Identical G: just a rotation, i.e. 1D interpolation
        if len(Gnew_to_Gms[i]) == 1:
            i_int = Gnew_to_Gms[i][0]
            sig_new[ind_sh,
                    :] = msinterp['interpolators'][i_int](newcyldotG)
            continue

        # Case where multiple reference shells are needed for interpolation
        Gms_ref = msinterp['Gms_un'][Gnew_to_Gms[i]]
        sig_ref_shells = np.zeros((Gms_ref.shape[0],
                                   newcyldotG.shape[0],
                                   num_subs))
        for j in range(Gms_ref.shape[0]):
            # First, interpolate at every shell of the multi-shell scheme
            i_int = Gnew_to_Gms[i][j]
            sig_ref_shells[j,
                           :,
                           :] = msinterp['interpolators'][i_int](newcyldotG)
        # Interpolate for new G value between shells, simultaneously for all
        # new directions:
        f_interp_G = interp1d(Gms_ref, sig_ref_shells,
                              axis=0,
                              kind='linear',
                              fill_value=np.nan,
                              assume_sorted=True)
        sig_new[ind_sh, :] = f_interp_G(G_un[i])
    return np.squeeze(sig_new)  # squeeze if num_subs=1


def init_PGSE_multishell_interp(sig_ms, sch_mat_ms, ordir):
    """Initializes multi-shell interpolator for faster interpolation.

    The output can be provided to interp_PGSE_from_multishell to make
    interpolation faster.

    Args:
      sig_ms: 2D NumPy array with shape (Nsampling, Nsubs) or a 1D array with
        shape (Nsampling,).
      sch_mat_ms: 2D NumPy array with shape (Nsampling, 7)
      ordir: NumPy array with 3 elements having unit Euclidean norm.

    Returns:
      interpolator object with precomputed settings which will make calls to
        interp_PGSE_from_multishell faster.
    """
    if ordir.size != 3:
        raise ValueError("Direction of dictionary computed with dense"
                         " sampling (ordir) should have 3 entries.")
    if ordir.ndim > 1:
        # Get rid of singleton dimensions because calls to 1D interpolators
        # below will be affected and dimension mismatch will occur
        ordir = np.squeeze(ordir)
    # Check inputs
    chk_ms = np.isclose(sch_mat_ms[0, 4:7], sch_mat_ms[:, 4:7])
    if not np.all(chk_ms):
        raise ValueError("Delta, delta and TE values should all be "
                         "identical in multi-shell sampling.")

    # Get inputs in right format
    if sig_ms.ndim == 1:
        sig_ms = sig_ms.reshape((sig_ms.size, 1))

    ordirnorm = np.sqrt((ordir**2).sum())  # (1,)
    if np.abs(1 - ordirnorm) > 1e-3:
        raise ValueError("Orientation vector of the multi-shell signal "
                         "must have unit norm. Detected"
                         " %g and %g respectively." % (ordirnorm,))
    # Check gradient norms of both schemes
    Gdir_norm_ms = np.sqrt(np.sum(sch_mat_ms[:, 0:3]**2, axis=1))
    if np.any(np.abs(1 - Gdir_norm_ms[Gdir_norm_ms > 0]) > 1e-3):
        raise ValueError("Gradient directions in multi-shell scheme matrix"
                         " should all either have zero or unit norm.")

    num_subs = sig_ms.shape[1]

    # Dot-products between fascicle directions and gradient directions
    orcyldotG_all = np.abs(np.dot(sch_mat_ms[:, 0:3], ordir))  # (Nmsh,)
    # Find unique G values
    Gms_un, i_Gms = np.unique(sch_mat_ms[:, 3], return_inverse=True)

    # Compute spherical interpolator for each shell of the dense sampling which
    # will be used for the new signals
    interpolators = []
    for i in range(Gms_un.shape[0]):
        # Get lines of the multi-shell dense sampling
        ind_sh = np.where(i_Gms == i)[0]  # (Ngrad_sh,)

        # If G==0, check that all G=0 signals identical in each column and do
        # not compute a spherical interpolator because it does not make sense
        if Gms_un[i] == 0:
            chk = np.all(np.isclose(sig_ms[ind_sh, :],
                                    sig_ms[ind_sh[0], :]),
                         axis=0)  # (Nsubs,)
            if np.any(~chk):
                bad_subs = np.where(~chk)[0]
                raise ValueError('Distinct signal values in provided multi-'
                                 'shell sampling for zero gradients, for '
                                 '%d substrate(s) [%s]' %
                                 (bad_subs.shape[0],
                                  " ".join("{:d}".format(b)
                                           for b in bad_subs)))
            # For b=0 (G=0) shells, an interpolator is just a constant
            # function. We just repeat, for each substrate, the first signal
            # value twice (in case only one b=0 acquisition present in
            # dense multi-shell sampling, which is likely for any efficient
            # sampling) for the x-axis points 0 and 1, the extreme values of
            # the dot product
            f_interp_shell = interp1d([0, 1],  # data below is (2, Nsubs) array
                                      np.repeat([sig_ms[ind_sh[0], :]],
                                                2, axis=0),
                                      axis=0,
                                      kind='linear',
                                      fill_value='extrapolate',
                                      assume_sorted=True)
            interpolators.append(f_interp_shell)
            continue
        # General G > 0 case in dense multi-shell sampling
        (sorted_orcyl_uni,
         sorted_ind_orcyl_uni) = np.unique(orcyldotG_all[ind_sh],
                                           return_index=True)
        dot_prod_data = sorted_orcyl_uni  # (Ngrad_sh,)
        sig_or_shell = sig_ms[ind_sh, :]  # (Ngrad_sh, Nsubs)
        sig_data = sig_or_shell[sorted_ind_orcyl_uni, :]  # (Ngrad_sh, Nsubs)

        # TODO: Add the data point (1, E_free*S0_b)

        # Smooth out data near dotproduct ~= 0, to avoid extrapolation
        # unstabilities due to MC variability with closely-spaced data :
        almost_perp = np.abs(dot_prod_data - dot_prod_data[0]) < 1e-3
        cluster_size = np.sum(almost_perp)

        # Substitute the x values close to left-most edge by their center of
        # mass. The if statement avoids taking the mean of an empty array,
        # which returns nan in Numpy (!). Do the same with the measurements.
        if cluster_size > 1:
            dot_prod_data = np.append(np.mean(dot_prod_data[almost_perp]),
                                      dot_prod_data[cluster_size:])
            sig_data = np.append(np.mean(sig_data[almost_perp, :],
                                         axis=0,
                                         keepdims=True),
                                 sig_data[cluster_size:, :],
                                 axis=0)  # ((Ngrad_sh-Nzeros), Nsubs)
        # Compute spherical interpolator of shell
        f_interp_shell = interp1d(dot_prod_data,
                                  sig_data,
                                  axis=0,
                                  kind='linear',
                                  fill_value='extrapolate',
                                  assume_sorted=True)
        interpolators.append(f_interp_shell)
    output = {'scheme_DeldelTE': sch_mat_ms[0, 4:7],
              'num_subs': num_subs,
              'Gms_un': Gms_un,
              'interpolators': interpolators}
    return output


def project_PGSE_scheme_xy_plane(sch_mat):
    """Removes gradients' component along z-axis

    Args:
      sch_mat: 2D NumPy array or string path to a text file with a one-line
        header. The first four entries of each row should be
        [gx, gy, gz, G] where [gx, gy, gz] is a unit-norm vector.

    Returns:
      The scheme matrix projected in the xy plane, i.e. with gz'=0, [gx', gy']
      having unit norm and G' such that (gz*G)**2 + G'**2 = G**2.
    """
    if isinstance(sch_mat, str):
        schemefile = sch_mat
        sch_mat = np.loadtxt(schemefile, skiprows=1)
    if sch_mat.ndim == 1:
        sch_mat = sch_mat[np.newaxis, :]
    # Proportion of gradient intensitiy in xy plane
    gxy = np.sqrt(sch_mat[:, 0]**2 + sch_mat[:, 1]**2)

    sch_mat_xy = np.zeros(sch_mat.shape)
    # Adjust gradient magnitudes in xy plane
    sch_mat_xy[:, 3] = sch_mat[:, 3] * gxy
    gxy[gxy == 0] = 1  # numerical stability before dividing
    # Normalize gradient directions to unit norm
    sch_mat_xy[:, :2] = sch_mat[:, :2]/gxy[:, np.newaxis]
    # Make sure zero gradients remain zero
    sch_mat_xy[sch_mat[:, 3] == 0, :4] = 0
    # Rest of protocol parameters remain unchanged (Del, del, TE)
    sch_mat_xy[:, 4:] = sch_mat[:, 4:]

    Gz = np.abs(sch_mat[:, 2]) * sch_mat[:, 3]
    G_chk_sq = sch_mat_xy[:, 3]**2 + Gz**2
    msg = ("Inconsistency with gradient intensities during"
           " projection in xy plane")
    assert np.all(np.abs(np.sqrt(G_chk_sq) - sch_mat[:, 3])
                  <= 1e-4 * sch_mat[:, 3]), msg  # keep <= for zeros !
    return sch_mat_xy


def import_PGSE_scheme(schemefile):
    """Import PGSE scheme file or matrix

    Args:
      schemefile: path to scheme file or NumPy array containing 7 entries per
        row.

    Returns:
      Always a 2D NumPy array with 7 entries per row.
    """
    if isinstance(schemefile, str):
        sch_mat = np.loadtxt(schemefile, skiprows=1)
    elif isinstance(schemefile, np.ndarray):
        sch_mat = schemefile
    else:
        raise TypeError("Unable to import a PGSE scheme matrix from input")
    if sch_mat.ndim == 1:
        # Return a 2D row matrix if only one sequence in protocol
        sch_mat = sch_mat[np.newaxis, :]
    if sch_mat.shape[1] != 7:
        raise RuntimeError("Detected %s instead of expected 7 colums in"
                           " PGSE scheme matrix." % sch_mat.shape[1])
    grad_norm = np.sqrt(np.sum(sch_mat[:, :3]**2, axis=1))
    num_bad_norms = np.sum(np.abs(1-grad_norm[grad_norm > 0]) > 1e-4)
    if num_bad_norms > 0:
        raise ValueError("Detected %d non-zero gradients which did not have"
                         " unit norm. Please normalize." % num_bad_norms)
    return sch_mat


def gen_SoS_MRI(S0, sigma_g, N=1):
    """Simulates Sum-of-Squares MRI signal for phased-array systems.

    Produces S_out = sqrt{ sum_{i=1}^N |S_i|^2 },
      where S_i = S_0 + eps1 + (1i)*eps2,
          with eps1, eps2 two independent zero-mean Gaussian variables of
          standard deviation sigma_g, assumed identical in all N coils, in
          both channels, and 1i the imaginary number.
    S_out follows a non-central Chi distribution.

    Args:
      S0: N-D NumPy array representing the true, possibly complex-valued,
        MRI contrast. Its entries represent acquisition parameters and/or
        multiple voxels.
      sigma_g: scalar or N-D NumPy array. Standard deviation of the Gaussian
        white noise in each coil, always assumed identical for all N coils
        and without inter-coil correlation.
        If `sigma_g` is a scalar, the standard deviation of the Gaussian noise
        is identical for all entries of `S0`.
        Else if `sigma_g.shape` is equal to `S0.shape`, then the standard
        deviation of the noise can be different for each entry of `S0`.
      N: the effective number of coils. Default is 1 (Rician noise).

    Returns:
      A scalar or NumPy array with the same shape as S0. The noise
      realizations are completely independent from one another.

    Raises:
      ValueError: if sigma_g is not a scalar but its shape does not match that
        of S0.
    """
    if np.all(sigma_g == 0):
        return np.sqrt(N)*S0  # perfect noiseless scenario

    if (np.ndim(sigma_g) > 0 and  # sigma_g is an array
            sigma_g.size > 1 and  # not a scalar
            S0.shape != sigma_g.shape):
        raise ValueError('sigma_g should either be a scalar or have '
                         'the shape (%s) of S0 for 1-to-1 '
                         'correspondance. Detected (%s) instead.'
                         % (", ".join("%d" % s for s in S0.shape),
                            ", ".join("%d" % s for s in sigma_g.shape)))

    Y = np.zeros(S0.shape, dtype=np.float64)
    for _ in range(N):
        noise_in_phase = sigma_g*np.random.randn(*S0.shape)
        noise_in_quadrature = sigma_g*np.random.randn(*S0.shape)
        Y = Y + (S0 + noise_in_phase)**2 + noise_in_quadrature**2
    # Pathological case when S0 has shape (N,) and sigma_g has shape (1, 1)
    # because due to NumPy's broadcasting rules Y will have shape (1, N)
    # instead of the desired (N,).
    return np.reshape(np.sqrt(Y), S0.shape)


#############################################################################
# DW-MRI VISUALIZATION
#############################################################################


def plot_multi_shell_signal(sig, sch_mat, fascdir,
                            plot_distr=None, substrate_names=None):
    """
    Args:
      sig: 1-D NumPy array with shape (num_seq,) or 2-D NumPy array with shape
        (num_seq, num_subs) where num_seq may be 1.
      sch_mat: 2-D NumPy array with num_seq rows and a least 6 columns,
        respectively representing g_x, g_y, g_z, G, Delta, delta.
        Alternatively, can be a string containing the path to a scheme file
        with one header row and at least 6 columns.
      fascdir: 1-D NumPy array with shape (3,) or 2-D array with shape (3, 1)
        to specify one common fascicle direction for all substrates.
        Use a 2-D NumPy array with shape (3, num_subs) to specify a different
        fascicle direction for each substrate.
      plot_distr: list of list where plot_distr[i] specifies the indices in
        [0, num_subs-1] of the substrates to plot on the i-th axes.
      substrate_names: list of strings specifiying the name of each substrate.
        len(substrate_names) must equal num_subs. If num_subs=1, it can be a
        simple string object.

    Raises:
      ValueError: if any of the above constraints is violated.
    """
    gam = get_gyromagnetic_ratio('H')  # 2*np.pi*42.577480e6

    # -- Check sig
    if np.ndim(sig) < 2:
        sig = np.reshape(sig, (sig.size, 1))
    num_subs = sig.shape[1]

    # -- Check sch_mat
    if isinstance(sch_mat, str):
        sch_mat = np.loadtxt(sch_mat, skiprows=1)
    if np.ndim(sch_mat) != 2:
        raise ValueError('Argument sch_mat should be a 2-D NumPy array, '
                         'detected %d dimensions.'
                         % np.ndim(sch_mat))
    if sch_mat.shape[1] < 6:
        raise ValueError('Argument sch_mat should be a 2-D array (or a '
                         'path to a file) containing at least 6 columns,'
                         ' detected %d columns here.'
                         % sch_mat.shape[1])
    if sch_mat.shape[0] != sig.shape[0]:
        raise ValueError('Argument sch_mat should have as many rows as '
                         'sig.shape[0] (=%d). Detected %d.'
                         % (sig.shape[0], sch_mat.shape[0]))
    Gdir_norm = np.sqrt(np.sum(sch_mat[:, :3]**2, axis=1))
    if np.any(~np.isclose(Gdir_norm[Gdir_norm > 0], 1)):
        raise ValueError('Argument sch_mat: the first three columns should'
                         'defined unit vectors or optionally zero vectors '
                         'for non diffusion-weighted signals.')

    # -- Check fascdir --
    if np.ndim(fascdir) < 2:
        fascdir = np.reshape(fascdir, (fascdir.shape[0], 1))
    if fascdir.shape[0] != 3:
        raise ValueError('Argument fascdir should contain column(s) with '
                         '3 elements, detected %d.' % fascdir.shape[0])
    orientations_norm = np.sqrt(np.sum(fascdir**2, axis=0))
    unit_orientations_ok = np.isclose(orientations_norm, 1)
    if np.any(~unit_orientations_ok):
        raise ValueError('Argument fascdir: fascicle direction(s) should be '
                         'unit column vector(s). Detected %d non-normalized'
                         ' orientation(s).'
                         % np.sum(~unit_orientations_ok))
    # Either one orientation for all or one orientation for each
    if fascdir.shape[1] == 1:
        # Use same principal direction for all single-fascicle substrates
        fascdir = np.tile(fascdir, (1, num_subs))
    else:
        if fascdir.shape[1] != num_subs:
            raise ValueError('Argument fascdir, if more than 1-column wide,'
                             ' should contain as many columns as there are '
                             'substrates to plot. Detected %d instead of %d.'
                             % (fascdir.shape[1], num_subs))
    # -- Check plot_distr
    if not plot_distr:  # None or empty
        plot_distr = list()
        for isub in range(num_subs):
            plot_distr.append([isub])
    num_axes = len(plot_distr)
    for i in range(num_axes):
        if np.any(np.array(plot_distr[i]) >= num_subs):
            raise ValueError('In plot_distr[%d], detected substrate '
                             'indice(s) equal to or exceeding num_subs-1'
                             ' (=%d).'
                             % (i, num_subs-1))

    # -- Check substrate_names
    subnames_are_IDs = False
    if substrate_names is None:
        substrate_names = list()
        subnames_are_IDs = True
        for isub in range(num_subs):
            substrate_names.append("sub %d" % (isub+1,))
    if isinstance(substrate_names, str):
        if num_subs == 1:
            substrate_names = [substrate_names]
        else:
            raise ValueError('Argument substrate_names should be a list of '
                             'strings and can only be a simple string if '
                             'num_subs=1, but detected num_subs=%d'
                             % num_subs)
    diff_time = sch_mat[:, 4]-sch_mat[:, 5]/3
    bvals = (gam*sch_mat[:, 3]*sch_mat[:, 5])**2 * diff_time
    bvals_un = np.unique(bvals)
    bcounts = np.zeros(bvals_un.shape)
    for ib in range(bvals_un.shape[0]):
        bcounts[ib] = np.sum(bvals == bvals_un[ib])
    # we only plot b-values containing at least 2 data points, unless it is a
    # b0 shell
    min_dpoints = 2
    bvals2plt = bvals_un[(bcounts >= min_dpoints) | (bvals_un == 0)]
    numbvals2plt = bvals2plt.size
    if numbvals2plt < bvals_un.size:
        print("WARNING: ignoring %d non-zero b-shell(s) containing less "
              "than %d data point(s). Plotting the other %d "
              "non-zero shells." %
              (bvals_un.size - numbvals2plt,
               min_dpoints,
               numbvals2plt - np.sum(bvals_un==0)))

    # Plot parameters
    m_sp_max = 2  # max number of subplot lines on one figure
    n_sp_max = 3  # max number of subplot columns on one figure
    markertypes = 'o+*xsd^v><ph'
    linestyles = ['-', '--', ':', '-.']
    linewidths = np.arange(0.4, 1.0, 0.05)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    lg_on_fig = False
    for iaxes in range(num_axes):
        subplotID = (iaxes % (m_sp_max*n_sp_max))
        if subplotID == 0:
            # New figure with m_sp x n_sp axes/subplots in it
            num_axes_left = num_axes - iaxes
            m_sp = min(m_sp_max, int(np.ceil(num_axes_left/n_sp_max)))
            n_sp = min(n_sp_max, int(np.ceil(num_axes_left/m_sp)))
            _, ax = plt.subplots(nrows=m_sp, ncols=n_sp, squeeze=False,
                                 sharex=True, sharey=True)
            lg_on_fig = False
        subID_to_plt = plot_distr[iaxes]
        ax_ix = int(np.floor((subplotID)/n_sp))
        ax_iy = (subplotID) % n_sp

        # Plotting on ax[ax_ix, ax_iy]
        for isub in range(len(subID_to_plt)):
            mtype = markertypes[isub % len(markertypes)]
            lstyle = linestyles[isub % len(linestyles)]
            lwidth = linewidths[isub % len(linewidths)]
            subID = subID_to_plt[isub]
            # Only show substrate ID if multiple substrates on current axes
            sub_ID_str = substrate_names[subID]
            for ish in range(numbvals2plt):
                bval = bvals2plt[ish]
                ind_sh = bvals == bval
                Gb = np.unique(sch_mat[ind_sh, 3])
                Delb = np.unique(sch_mat[ind_sh, 4])
                delb = np.unique(sch_mat[ind_sh, 5])

                assert Gb.size == 1, "Internal problem Gb shell %d" % ish
                assert Delb.size == 1, "Internal problem Delb shell %d" % ish
                assert delb.size == 1, "Internal problem delb shell %d" % ish

                # Reorder signal according to |g.n|
                sig_shell = sig[ind_sh, subID]
                Gdir = sch_mat[ind_sh, :3]
                # abs dot product gradient_dir*original_cyl_dir
                cyldotG = np.abs(Gdir @ fascdir[:, subID])  # Ngrad_sh
                sorted_ind_dp = np.argsort(cyldotG)
                sorted_dp = cyldotG[sorted_ind_dp]

                shell_str = ('b=%d G=%3.1f D=%g d=%g' %
                             (np.round(bval/1e6), Gb*1e3, Delb*1e3, delb*1e3))
                if isub > 0:
                    # only show shell info for first plotted subtrate
                    shell_str = ''
                # Line visible if 1st shell of new substrate, 1st substrate of
                # new shell
                disp_str = '_nolegend_'
                if isub == 0 or ish == 0:
                    if len(subID_to_plt) > 1:
                        disp_str = sub_ID_str + ' ' + shell_str
                    else:
                        disp_str = shell_str
                ax[ax_ix, ax_iy].plot(sorted_dp,
                                      sig_shell[sorted_ind_dp],
                                      linestyle=lstyle,
                                      linewidth=lwidth,
                                      marker=mtype,
                                      fillstyle='none',
                                      color='C%d' % (ish % 10,),
                                      label=disp_str)
            # end for ish
        # end for isub
        handles, labels = ax[ax_ix, ax_iy].get_legend_handles_labels()

        ax[ax_ix, ax_iy].set_xlabel('|g.n|')
        ax[ax_ix, ax_iy].set_ylabel('S')

        # Show legend if needed, set title
        if len(subID_to_plt) > 1:
            # show legend if multiple substrates are plotted
            ax[ax_ix, ax_iy].legend(handles,
                                    labels,
                                    loc='northeast')
            # (matplotlib.axes.Axes.legend)
            lg_on_fig = True
        else:
            # only one substrate to plot but still convenient to have
            # substrate ID or name
            if subnames_are_IDs:
                ax[ax_ix, ax_iy].set_title('Sub. %d' % subID_to_plt[0],
                                           fontdict={'fontweight': 'bold'})
            else:
                ax[ax_ix, ax_iy].set_title(sub_ID_str,
                                           fontdict={'fontweight': 'bold'})
            if (~lg_on_fig and
                    ((subplotID+1) == m_sp*n_sp or
                     (subplotID+1) == num_axes)):
                # Show legend if last axes of the figure and still no legend
                # on the current figure
                ax[ax_ix, ax_iy].legend(handles,
                                        labels,
                                        loc='best')
                # should be useless bc next axes should be on a new figure,
                # which will trigger lg_on_fig=False. Kept anyway for
                # debugging:
                lg_on_fig = True
    # end for iaxes


def plot_signal_2Dprotocol(sig, sch_mat, display_names=None):
    """Plots signals for an AxCaliber-like acquisition protocol.

    Splits the data per (Delta, delta) pair, plots 3 pairs per figure. All
    substrates are plotted on the same axes.

    Args:
      sig: 1-D or 2-D NumPy array of shape (Nsigs,) or (Nsigs, Nsubs)
      sch_mat: 2-D NumPy array of shape (Nsig, 6) or (Nsig, 7)

    """
    # Check input
    if np.any(sch_mat[:, 2] != 0):
        raise ValueError("Use the original schemefile with zeros for gz")
    if sig.ndim == 1:
        # raise ValueError("Plot one signal at a time: sig should be 1D")
        sig = sig[:, np.newaxis]
    elif sig.ndim > 2:
        raise ValueError("sig should be a 1D or 2D NumPy array, detected"
                         " %d dimensions." % sig.ndim)
    if sig.shape[0] != sch_mat.shape[0]:
        raise ValueError("Mismatch: detected %d values in signal(s) to plot"
                         "versus %d sequences in acquisition protocol."
                         % (sig.shape[0], sch_mat.shape[0]))
    numsubs = sig.shape[1]

    # Substrate display names
    if display_names is None:
        display_names = []
        for isub in range(numsubs):
            display_names.append("sub %d " % (isub,))

    # Set number of 'acquisition lines', i.e. direction irrespective of
    # polarity
    num_lines = 2

    # Set number of figures (i.e., number of Del/del pairs) per figure
    fig_per_plt = 3

    # One line style per line in protocol
    lstyles = ['-', '--', '-.', ':']
    # https://matplotlib.org/examples/color/colormaps_reference.html
    ncolors = 8
    cmap = plt.get_cmap('Set1')(np.linspace(0, 1, ncolors))

    # Analyze PGSE protocol
    Gxy = sch_mat[:, 3]  # here it works since gz=0 everywhere
    gdir_xy = sch_mat[:, :3]  # includes gz always = 0

    # Extract unique delta pairs
    Deldel_un, i_un = np.unique(sch_mat[:, 4:6],
                                return_inverse=True,
                                axis=0)
    num_Deldels = Deldel_un.shape[0]

    # Max y-value across all plots
    max_sig = np.max(sig)

    for idel in range(num_Deldels):
        if idel % fig_per_plt == 0:
            fig, ax = plt.subplots(1, fig_per_plt, sharey=True)
            ax[idel % fig_per_plt].set_ylim(top=max_sig)
        ind_del = np.where(i_un == idel)[0]
        gdir_un, ig_un = np.unique(sch_mat[ind_del, :3],
                                   return_inverse=True,
                                   axis=0)
        assert (gdir_un.shape[0] == (2*num_lines
                + 1)), ("Problem at delta pair %d/%d: found %d instead of %d "
                        "unique gradient directions in scheme "
                        "matrix(including b0 sequences)." %
                        (idel+1, num_Deldels,
                         gdir_un.shape[0],
                         2*num_lines + 1))
        # Cross dot products will have a diagonal of 1s, some off-diagonal
        # elements equal to -1 (if two opposite directions are present) and
        # some others close to zero. This allows us to find pairs of opposite
        # directions
        cross_dp = gdir_un @ gdir_un.T
        ig, ig_op = np.where(np.isclose(cross_dp, -1))
        is_upper = ig < ig_op  # upper triangular part of matrix
        ig = ig[is_upper]
        ig_op = ig_op[is_upper]
        assert (ig.size == 2), ("Problem at delta pair %d/%d: found %d "
                                "instead of 2 pairs of opposite directions"
                                " in scheme matrix." %
                                (idel+1, num_Deldels, ig.size))
        # Plot b0 sequences
        label_deltas = "Del=%gs del=%gs" % (Deldel_un[idel, 0]*1e3,
                                            Deldel_un[idel, 1]*1e3)
        for isub in range(numsubs):
            sig_b0_delpair = sig[ind_del, isub][Gxy[ind_del] == 0]
            label = None
            if isub == 0:
                label = 'b0 (%s)' % label_deltas
            col = cmap[isub % len(cmap)]
            ax[idel % fig_per_plt].plot(np.zeros(sig_b0_delpair.shape),
                                        sig_b0_delpair,
                                        marker='x', color=col,
                                        label=label)
        # Plot signal along each line, for each substrate
        for isub in range(numsubs):
            sublabel = display_names[isub]
            col = cmap[isub % len(cmap)]
            # Redo computation of signed gradient (not the most efficient
            # but makes for better visualization)
            for idir in range(ig.size):
                linedir = gdir_un[ig[idir]]
                indline = ind_del[(ig_un == ig[idir])
                                  | (ig_un == ig_op[idir])]  # either polarity
                G_signed = (Gxy[indline] *
                            np.sign(gdir_xy[indline, :] @ linedir))
                ls = lstyles[idir % len(lstyles)]
                label = None  # no legend entry by default
                linelabel = "[" + " ".join("%.3f" % e for e in linedir) + "]"
                if isub == 0:
                    # Label each line once
                    label = " " * (len(sublabel) + 4) + linelabel
                    if idir == 0:
                        label = sublabel + " " + linelabel
                elif idir == 0:
                    # Label each substrate once
                    label = sublabel
                # Plot line signal
                ax[idel % fig_per_plt].plot(G_signed,
                                            sig[indline, isub],
                                            marker='.',
                                            color=col,
                                            linestyle=ls,
                                            label=label)
        ax[idel % fig_per_plt].legend()
        ax[idel % fig_per_plt].grid()
        ax[idel % fig_per_plt].set_xlabel("signed G")


#############################################################################
# I/0
#############################################################################


# Code by cs01on posted on stackoverflow #7008608
# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
# Handles Matlab structures which may themselves includes cell array of objects
def loadmat(filename):
    '''
    This function should be called instead of scipy.io.loadmat
    as it solves the problem of not properly recovering Python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif (str(d[key].__class__) ==
                  "<class 'scipy.io.matlab.mio5_params.mat_struct'>"):
                # ugly hack for Spyder 3.3.2 conda 4.5.12 on CentOS 7
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested
        dictionaries
        '''
        d = {}
        for strg in matobj.__dict__.keys():  # vs in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif (str(elem.__class__) ==
                  "<class 'scipy.io.matlab.mio5_params.mat_struct'>"):
                # ugly hack for Spyder 3.3.2 conda 4.5.12 on CentOS 7
                d[strg] = _todict(elem)
            # elif isinstance(elem, np.ndarray):
            #    d[strg] = _tolist(elem)  # GR: not sure why that's useful?
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as NumPy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif (str(sub_elem.__class__) ==
                  "<class 'scipy.io.matlab.mio5_params.mat_struct'>"):
                # ugly hack for Spyder 3.3.2 conda 4.5.12 on CentOS 7
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)
