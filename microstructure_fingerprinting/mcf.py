# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:32:29 2020

Python implementation of the multiple-correlation matrix formalism (MCF)
based on Denis Grebenkov's Matlab implementation and Gaetan Rensonnet's
revised version specific to the PGSE sequence.

TODO: write loader function to fetch data and return Lam, Lamvec and B

@author: rensonnetg
"""
import numpy as np
import os
from scipy.linalg import expm
try:
    # package added to Python environment
    from . import mf_utils as mfu
except ImportError:
    # local, occasional use
    import mf_utils as mfu


def import_DDE_scheme(schemefile):
    """Import DDE scheme file or matrix

    Args:
      schemefile: path to scheme file or NumPy array containing 14 entries per
        row.

    Returns:
      Always a 2D NumPy array with 7 entries per row.
    """
    if isinstance(schemefile, str):
        sch_mat = np.loadtxt(schemefile, skiprows=1)
    elif isinstance(schemefile, np.ndarray):
        sch_mat = schemefile
    else:
        raise TypeError("Unable to import a DDE scheme matrix from input")
    if sch_mat.ndim == 1:
        # Return a 2D row matrix if only one sequence in protocol
        sch_mat = sch_mat[np.newaxis, :]
    if sch_mat.shape[1] != 14:
        raise RuntimeError("Detected %s instead of expected 14 colums in"
                           " PGSE scheme matrix." % sch_mat.shape[1])
    grad_norm1 = np.sqrt(np.sum(sch_mat[:, :3]**2, axis=1))
    num_bad_norms1 = np.sum(np.abs(1-grad_norm1[grad_norm1 > 0]) > 1e-4)
    if num_bad_norms1 > 0:
        raise ValueError("Detected %d non-zero gradients in the first "
                         "encoding module which did not have"
                         " unit norm. Please normalize." % num_bad_norms1)
    grad_norm2 = np.sqrt(np.sum(sch_mat[:, 7:10]**2, axis=1))
    num_bad_norms2 = np.sum(np.abs(1-grad_norm2[grad_norm2 > 0]) > 1e-4)
    if num_bad_norms2 > 0:
        raise ValueError("Detected %d non-zero gradients in the second "
                         "encoding module which did not have"
                         " unit norm. Please normalize." % num_bad_norms2)
    Del1 = sch_mat[:, 4]
    del1 = sch_mat[:, 5]
    Del2 = sch_mat[:, 11]
    del2 = sch_mat[:, 12]
    tau_mix = sch_mat[:, 6]
    TE = sch_mat[:, 13]
    T = Del1 + del1 + tau_mix + Del2 + del2

    n_bad_del1 = np.sum(Del1 < del1)
    if n_bad_del1 > 0:
        raise ValueError("Detected %d sequences in first encoding module"
                         " where gradient separation Delta was less than"
                         " gradient duration delta." % n_bad_del1)
    n_bad_del2 = np.sum(Del2 < del2)
    if n_bad_del2 > 0:
        raise ValueError("Detected %d sequences in second encoding module"
                         " where gradient separation Delta was less than"
                         " gradient duration delta." % n_bad_del2)
    n_bad_T = np.sum(T > TE)
    if n_bad_T > 0:
        raise ValueError("Detected %d sequences in which the total "
                         "diffusion time (Delta1+delta1+tau_mix+Delta2"
                         "+delta2) exceeded the echo time TE." % n_bad_T)
    return sch_mat


def MCF_DDE(domain, L, diff, scheme,
            envdir=np.array([0, 0, 1]),
            gamma=mfu.get_gyromagnetic_ratio('hydrogen'),
            M=60):
    ''' Thu Apr 16 20:04:39 2020
    '''
    sch_mat = import_DDE_scheme(scheme)  # always 2D
    n_seq = sch_mat.shape[0]
    gdirall1 = sch_mat[:, :3]
    Gall1 = sch_mat[:, 3]
    Delall1 = sch_mat[:, 4]
    delall1 = sch_mat[:, 5]
    tauall = sch_mat[:, 6]

    gdirall2 = sch_mat[:, 7:10]
    Gall2 = sch_mat[:, 10]
    Delall2 = sch_mat[:, 11]
    delall2 = sch_mat[:, 12]

    Tall = Delall1 + delall1 + tauall + Delall2 + delall2
    Tmax = np.max(Tall)  # just for scaling

    # Normalize environment's direction
    dir_norm = np.sqrt(np.sum(envdir**2))
    if dir_norm == 0:
        raise ValueError('Direction (orientation) of environment cannot'
                         ' be a zero vector.')
    else:
        envdir = envdir/dir_norm

    # Number of eigenfunctions of the Laplace operator in the geometry used
    # for the decomposition of the macroscopic magnetization
    M = np.min([M, 60])

    # Load pre-computed MCF vectors and matrices
    if domain in ['c', 'cylinder']:  # TODO: 'fc', 'fincyl'
        DOMTYPE = 'cylinder'
        fname = 'MCF_Bcl.mat'
    elif domain in ['s', 'sphere']:
        DOMTYPE = 'sphere'
        fname = 'MCF_Bsl.mat'
    elif domain in ['p', 'planes']:
        DOMTYPE = 'interval'
        fname = 'MCF_Bpl.mat'
    else:
        raise ValueError('Unknown domain %s.' % (domain,))

    # 'B' matrix, perturbing interaction due to the externally-applied
    # magnetic field (via gradients) in the eigenbasis of the Laplace
    # operator. Eq. [22] in [1].
    script_dir = os.path.dirname(__file__)
    MCF_dir = os.path.join(script_dir, 'MCF_data')
    path_B = os.path.join(MCF_dir, fname)
    B = mfu.loadmat(path_B)['B'][0:M, 0:M]

    # Eigenvalues of the Laplace operator, which are always real and
    # positive and strictly increasing (lam_1<lam_2<...<lam_M<...). Eq. [14a]
    # in [1].
    fname = list(fname)  # convert to list for modification of a character
    fname[4] = 'L'
    path_L = os.path.join(MCF_dir, "".join(fname))
    Lamvec = mfu.loadmat(path_L)['L'].squeeze()[0:M]
    Lam = np.diag(Lamvec)

    # ! Check accuracy of truncation of B matrix and Lambda vector as per
    # Eq. [36] in [1]. TODO: double-check for double-diffusion encoding
    # dimensionless number, pure diffusion, 'damping' real factor in matrix
    # exponential:
    p = diff*Tall/L**2
    # dimensionless number, effect of gradient encoding, 'oscillating'
    # imaginary factor in matrix exponential:
    Gallmax = np.maximum(Gall1, Gall2)  # element-wise maximum
    q = gamma * Tall * L * Gallmax
    # ratio, operations reordered for numerical stability:
    q_over_p = (gamma*L)*(L**2/diff)*Gallmax
    idx_bad = np.where(q_over_p >= Lamvec[-1])[0]
    n_bad = idx_bad.size
    if n_bad > 0:
        badlist = " ".join("%d" % (idx,) for idx in idx_bad)
        bad1 = idx_bad[0]
        # p*Lam = damping factor, which must exceed oscillating component q
        msg = ('Number of eigenvalues and eigenfunctions M=%d too small'
               ' to ensure accuracy of final DW-MRI signal for the'
               ' physical parameters provided in the following %d '
               'sequence(s):\n%s\n'
               'In seq. %d for instance, detected lambda_M=%g <'
               ' q/p=%g, with p=D*T/L^2=%g and q=gam*T*L*G=%g.'
               'This code is only reliable for a smaller L, a smaller G '
               'or a higher diff. Otherwise you may want to try to '
               'approximate the DW-MRI signal with a formula based '
               'on the Gaussian phase distribution (GPD) for instance.' %
               (M, n_bad, badlist, bad1, Lamvec[-1],
                q_over_p[bad1], p[bad1], q[bad1]))
        raise ValueError(msg)

    # Precompute component associated to pure diffusion (just needs to be
    # scaled accordingly for each sequence afterwards)
    Epurediff = np.exp(-Lamvec*diff*Tmax/L**2)

    # Compute normalized signal attenuation for all sequences
    E = np.zeros(n_seq)
    for i in range(n_seq):
        G1 = Gall1[i]
        G2 = Gall2[i]
        gdir1 = gdirall1[i, :]
        gdir2 = gdirall2[i, :]
        Del1 = Delall1[i]
        del1 = delall1[i]
        Del2 = Delall2[i]
        del2 = delall2[i]
        tau = tauall[i]
        T_i = Del1 + del1 + tau + Del2 + del2
        if G1 == 0 and G2 == 0:
            E[i] = 1
            continue

        if DOMTYPE == 'cylinder':
            gpar1 = np.dot(gdir1, envdir) * envdir
            gperp1 = gdir1 - gpar1
            Gpar1 = G1 * np.sqrt(np.sum(gpar1**2))
            Gperp1 = G1 * np.sqrt(np.sum(gperp1**2))

            gpar2 = np.dot(gdir2, envdir) * envdir
            gperp2 = gdir2 - gpar2
            Gpar2 = G2 * np.sqrt(np.sum(gpar2**2))
            Gperp2 = G2 * np.sqrt(np.sum(gperp2**2))

            # Signal due to component of gradient perpendicular to cylinder
            p = diff * T_i / L**2
            qperp1 = gamma * T_i * L * Gperp1
            qperp2 = gamma * T_i * L * Gperp2
            # TODO: shortcut if G1=0 => R1=Identity, same for G2 and R2
            R1 = (expm(-(p*Lam - (1j)*qperp1*B)*(del1/T_i)) @
                  np.diag(Epurediff**((Del1-del1)/Tmax)) @
                  expm(-(p*Lam + (1j)*qperp1*B)*(del1/T_i)))
            Rmix = np.diag(Epurediff**(tau/Tmax))
            R2 = (expm(-(p*Lam - (1j)*qperp2*B)*(del2/T_i)) @
                  np.diag(Epurediff**((Del2-del2)/Tmax)) @
                  expm(-(p*Lam + (1j)*qperp2*B)*(del2/T_i)))
            R = R2 @ Rmix @ R1
            Eperp = R[0, 0]
            # Signal due to component of gradient parallel to cylinder
            bpar1 = (gamma * del1 * Gpar1)**2 * (Del1 - del1/3)
            bpar2 = (gamma * del2 * Gpar2)**2 * (Del2 - del2/3)
            Epar = np.exp(-(bpar1+bpar2) * diff)

            # Total signal for applied gradient profile i:
            E[i] = np.abs(Eperp) * Epar
        else:
            raise NotImplementedError()
    return E


def MCF_PGSE(domain, L, diff, *,  # all subsequent args must be named
             scheme=None, envdir=np.array([0, 0, 1]),
             G=None, Delta=None, delta=None,
             L2=None,
             gamma=mfu.get_gyromagnetic_ratio('hydrogen'), M=60):
    '''Intracellular PGSE signal attenuation using the MCF approach.

    The multiple correlation function (MCF) formalism [1] provides a very
    accurate value for the DW-MRI signal inside simple geometries in which
    the Laplace eigenvalue problem has a known solution, for gradient profiles
    that are piecewise constant. Multiple boundary conditions can be handled
    (Dirichlet, Neumann, Robin) but only the Neumann condition representing
    perfectly-reflecting membranes is implemented here. The accuracy of the
    solution is determined by the number of Laplace eigenfunctions used to
    represent the solution, which is limited to 60 in the current
    implementation. If gamma*G*L^3/Diff is too large, the validity of the
    current implementation breaks and an error message is displayed.

    Args:
      domain: (str) 'c', 'cylinder' for infinite cylinder; 'fc' or 'fincyl'
        for finite cylinders (L2 required); 's' or 'sphere' for sphere;
        'p' or 'planes' for a 1D interval or, equivalently, the space between
        two infinite 2D planes or slabs.
      L: (scalar) size of geometrical pore ('characteristic length' in MCF
        theory), namely radius for cylinder, length for the finite cylinder,
        radius for the sphere and length of interval for plane
      (required: specificy either scheme or all of G, Delta, delta)
      scheme: path to scheme file or NumPy array containing 7 entries per
        row: [gx, gy, gz, G, Delta, delta, TE]. If set, G, Delta and delta
        are ignored.
      G: scalar or 1D NumPy array. Same shape as Delta and delta. Ignored if
        scheme is set.
      Delta: scalar or 1D NumPy array. Same shape as G and delta.
      delta: scalar or 1D NumPy array. Same shape as G and Delta.

    References:
      [1] Grebenkov, D.S., 2008. Laplacian eigenfunctions in NMR. I. A
      numerical tool. Concepts in Magnetic Resonance Part A: An Educational
      Journal, 32(4), pp.277-301.
    '''
    if scheme is not None:
        sch_mat = mfu.import_PGSE_scheme(scheme)  # always 2D
        n_seq = sch_mat.shape[0]
        gdirall = sch_mat[:, :3]
        Gall = sch_mat[:, 3]
        Delall = sch_mat[:, 4]
        delall = sch_mat[:, 5]
        Tmax = np.max(Delall + delall)
    else:
        all_missing = ((G is None) and (Delta is None) and (delta is None))
        if all_missing:
            raise ValueError('Either provide a scheme matrix or specify'
                             ' G, Delta and delta.')
        missing = ((G is None) or (Delta is None) or (delta is None))
        if missing:
            raise ValueError('Without a scheme matrix provided (non-scheme'
                             ' mode), G, Delta and delta are all required.')
        Gall = np.atleast_1d(G)
        Delall = np.atleast_1d(Delta)
        delall = np.atleast_1d(delta)
        samesize = (Gall.size == Delall.size) and (Delall.size == delall.size)
        if not samesize:
            raise ValueError('G, Delta and delta should contain the same'
                             ' number of elements. Detected %d, %d and '
                             '%d, respectively.' %
                             (Gall.size, Delall.size, delall.size))
        n_seq = Gall.size
        Tmax = np.max(Delall + delall)
        # In non-scheme mode, environment assumed oriented along z axis and
        # gradient assumed along x-axis if none is provided
        ref_gdir = np.array([1, 0, 0])
        gdirall = np.tile(ref_gdir, (n_seq, 1))  # make 2D
    Tall = Delall + delall
    n_bad_del = np.sum(Delall < delall)
    if n_bad_del > 0:
        # TODO: transfer this repsonsibility to importation script
        raise ValueError('Detected %d sequence(s) with big Delta smaller'
                         ' than small delta. In a PGSE sequence, Delta>=delta'
                         ' should always be enforced.' % (n_bad_del,))
    # Normalize environment's direction
    dir_norm = np.sqrt(np.sum(envdir**2))
    if dir_norm == 0:
        raise ValueError('Direction (orientation) of environment cannot'
                         ' be a zero vector.')
    else:
        envdir = envdir/dir_norm

    # Number of eigenfunctions of the Laplace operator in the geometry used
    # for the decomposition of the macroscopic magnetization
    M = np.min([M, 60])

    # Load pre-computed MCF vectors and matrices
    if domain in ['c', 'cylinder']:  # TODO: 'fc', 'fincyl'
        DOMTYPE = 'cylinder'
        fname = 'MCF_Bcl.mat'
    elif domain in ['s', 'sphere']:
        DOMTYPE = 'sphere'
        fname = 'MCF_Bsl.mat'
    elif domain in ['p', 'planes']:
        DOMTYPE = 'interval'
        fname = 'MCF_Bpl.mat'
    else:
        raise ValueError('Unknown domain %s.' % (domain,))

    # 'B' matrix, perturbing interaction due to the externally-applied
    # magnetic field (via gradients) in the eigenbasis of the Laplace
    # operator. Eq. [22] in [1].
    script_dir = os.path.dirname(__file__)
    MCF_dir = os.path.join(script_dir, 'MCF_data')
    path_B = os.path.join(MCF_dir, fname)
    B = mfu.loadmat(path_B)['B'][0:M, 0:M]

    # Eigenvalues of the Laplace operator, which are always real and
    # positive and strictly increasing (lam_1<lam_2<...<lam_M<...). Eq. [14a]
    # in [1].
    fname = list(fname)  # convert to list for modification of a character
    fname[4] = 'L'
    path_L = os.path.join(MCF_dir, "".join(fname))
    Lamvec = mfu.loadmat(path_L)['L'].squeeze()[0:M]
    Lam = np.diag(Lamvec)

    # ! Check accuracy of truncation of B matrix and Lambda vector as per
    # Eq. [36] in [1].
    # dimensionless number, pure diffusion, 'damping' real factor in matrix
    # exponential:
    p = diff*Tall/L**2
    # dimensionless number, effect of gradient encoding, 'oscillating'
    # imaginary factor in matrix exponential:
    q = gamma * Tall * L * Gall
    # ratio, operations reordered for numerical stability:
    q_over_p = (gamma*L)*(L**2/diff)*Gall
    idx_bad = np.where(q_over_p >= Lamvec[-1])[0]
    n_bad = idx_bad.size
    if n_bad > 0:
        badlist = " ".join("%d" % (idx,) for idx in idx_bad)
        bad1 = idx_bad[0]
        # p*Lam = damping factor, which must exceed oscillating component q
        msg = ('Number of eigenvalues and eigenfunctions M=%d too small'
               ' to ensure accuracy of final DW-MRI signal for the'
               ' physical parameters provided in the following %d '
               'sequence(s):\n%s\n'
               'In seq. %d for instance, detected lambda_M=%g <'
               ' q/p=%g, with p=D*T/L^2=%g and q=gam*T*L*G=%g.'
               'This code is only reliable for a smaller L, a smaller G '
               'or a higher diff. Otherwise you may want to try to '
               'approximate the DW-MRI signal with a formula based '
               'on the Gaussian phase distribution (GPD) for instance.' %
               (M, n_bad, badlist, bad1, Lamvec[-1],
                q_over_p[bad1], p[bad1], q[bad1]))
        raise ValueError(msg)

    # Precompute component associated to pure diffusion (just needs to be
    # scaled accordingly for each sequence afterwards)
    Epurediff = np.exp(-Lamvec*diff*Tmax/L**2)

    # Compute normalized signal attenuation for all sequences
    E = np.zeros(n_seq)
    for i in range(n_seq):
        G = Gall[i]
        if G == 0:
            E[i] = 1
            continue
        gdir = gdirall[i, :]
        gdirnorm = np.sqrt(np.sum(gdir**2))
        if np.abs(1-gdirnorm) > 1e-4:
            raise ValueError('Sequence %d: gradient direction not normalized'
                             ' (found %g)' % (i, gdirnorm))
        Del_i = Delall[i]
        del_i = delall[i]
        T_i = Del_i + del_i

        if DOMTYPE == 'cylinder':
            gpar = np.dot(gdir, envdir) * envdir
            gperp = gdir - gpar
            Gpar = G * np.sqrt(np.sum(gpar**2))
            Gperp = G * np.sqrt(np.sum(gperp**2))
            # Signal due to component of gradient perpendicular to cylinder
            p = diff * T_i / L**2
            qperp = gamma * T_i * L * Gperp
            R = (expm(-(p*Lam - (1j)*qperp*B)*(del_i/T_i)) @
                 np.diag(Epurediff**((Del_i-del_i)/Tmax)) @
                 expm(-(p*Lam + (1j)*qperp*B)*(del_i/T_i)))
            Eperp = R[0, 0]
            # Signal due to component of gradient parallel to cylinder
            bpar = (gamma * del_i * Gpar)**2 * (Del_i - del_i/3)
            Epar = np.exp(-bpar * diff)
            E[i] = np.abs(Eperp) * Epar
        else:
            raise NotImplementedError()
    return E
