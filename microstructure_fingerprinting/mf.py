# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:34:12 2020

User interface for microstructure fingerprinting following Dipy's style.

@author: rensonnetg
"""
try:
    # package added to Python environment
    from . import mf_utils as mfu
except ImportError:
    # local, occasional use
    import mf_utils as mfu

import multiprocessing as mp
import nibabel as nib
import numpy as np
import os
import time


def cleanup_2fascicles(frac1, frac2, peakmode,
                       mu1, mu2, mask, frac12=None):
    """Cleans up detected fascicle orientations (or "peaks").

    Applies the method described in [1] which selects 0, 1 or 2 of the 2
    detected peaks and updates their orientations based on the detected
    fascicles' weights and orientations. The goal is to get rid of spurious
    peaks.

    Parameters
    ----------
    frac1: str or NumPy array
        normalized weight of first fascicle in [0, 1]
    frac2: str or NumPy array
        normalized weight of second fascicle in [0, 1]. Note that frac1 and
        frac2 need not sum to 1 in every voxel.
    peakmode: str
        One of 'colat_longit', 'tensor' or 'peaks' (see below).
    mu1: str or NumPy array
        Orientation of fascicle 1 in each voxel. Depending on peakmode, the
        last dimension of the array is assumed to contain
          'colat_longit': colatitute (from positive z-axis) and longitude
          (from positive x-axis) angles; mu1.shape[-1] == 2.
          'peaks': x, y, and z coordinate; mu1.shape[-1] == 3.
          'tensor': Dxx, Dxy, Dxz, Dyy, Dyz, Dzz elements of the diffusion
           tensor ('row order'); mu1.shape[-1] == 6.
    mu2: str or NumPy array
        orientation of the second fascicle, see mu1.
    mask: str or NumPy array
        brain mask indicating where fascicles need cleaning up.
    frac12: str or NumPy array (optional)
        Fractions of fascicles 1 and 2 in the same array. If specified, takes
        precedence over frac1 and frac2.

    Returns
    ---------
    peaks_out: NumPy array with ndim = frac1.ndim + 1 and with shape[-1]=6. In
        each data voxel, the first three entries represent the orientation of
        the first fascicle, the last three of the second fascicle. Entries
        set to zero if no fascicle detected.
    num_fasc_out: NumPy array with same shape as frac1. Contains the number
        detected fascicles in each voxel.

    References
    ----------
    .. [1] Rensonnet, G., Scherrer, B., Girard, G., Jankovski, A.,
    Warfield, S.K., Macq, B., Thiran, J.P. and Taquet, M., 2019. Towards
    microstructure fingerprinting: Estimation of tissue properties from
    a dictionary of Monte Carlo diffusion MRI simulations. NeuroImage,
    184, pp.964-980.
    """
    # if ratio of large fasc over small fasc is more than this, small fasc is
    # discarded unless its relative weight exceeded the keep threshold
    ratio = 3.0
    # relative weight above which no fascicle can ever be discarded
    w_keep = 0.18
    # relative weight under which a fascicle is discarded
    w_small = 0.10
    # crossing angle under which 2 orientations are merged [deg]
    ang_min = 15

    if frac1 is None or frac2 is None:
        msg = ("If fractions of first and second fascicles set to None,"
               " argument frac12 is required to specify both fractions"
               " simultanously. A total of 6 arguments should be passed,"
               " not 5.")
        if frac12 is None:
            raise ValueError(msg)

    if isinstance(mask, str):
        mask = nib.load(mask).get_data()
    if isinstance(frac1, str):
        frac1 = nib.load(frac1).get_data()
    if isinstance(frac2, str):
        frac2 = nib.load(frac2).get_data()

    if frac12 is not None:
        if isinstance(frac12, str):
            frac12 = nib.load(frac12).get_fdata()
        if frac12.shape[-1] < 2:
            raise ValueError("Last dimension of frac12 should have size"
                             " at least 2.")
        # Overwrites frac1 and frac2 if those had been set
        if frac12.shape[mask.ndim] == 1:
            # data has shape (Nx, Ny, Nz, 1, 2) => get rid of singleton dim
            frac1 = frac12[..., 0, 0]
            frac2 = frac12[..., 0, 1]
        else:
            frac1 = frac12[..., 0]
            frac2 = frac12[..., 1]

    if isinstance(mu1, str):
        mu1 = nib.load(mu1).get_data()
    if isinstance(mu2, str):
        mu2 = nib.load(mu2).get_data()

    if frac1.shape != mask.shape:
        raise ValueError("frac1 should have the same shape as mask")
    if frac2.shape != mask.shape:
        raise ValueError("frac2 should have the same shape as mask")

    # Check shape of peak arguments
    if peakmode == 'colat_longit':
        lastdimsize = 2
    elif peakmode == 'peaks':
        lastdimsize = 3
    elif peakmode == 'tensor':
        lastdimsize = 6
        # shape of tensor file data is often (nx, ny, nz, 1, 6)
        # Get rid of singleton dimension in next-to-last axis
        if mu1.shape[mask.ndim] == 1:
            mu1 = mu1[..., 0, :]
        if mu2.shape[mask.ndim] == 1:
            mu2 = mu2[..., 0, :]
    else:
        raise ValueError('Unknown peak mode %s' % peakmode)

    # Check shape
    if mu1.shape[-1] != lastdimsize or mu1.shape[-1] != lastdimsize:
            msg = ('In \'%s\' peak mode, last dimension of mu1 and mu2 '
                   'should have size %d. Detected %d and %d.'
                   % (lastdimsize, mu1.shape[-1], mu2.shape[-1]))

#    peakmode = ''
#    if mu1.shape[-1] == 2 and mu2.shape[-1] == 2:
#        peakmode = 'colat_longit'
#    elif mu1.shape[-1] == 6 and mu1.shape[-1] == 6:
#        peakmode = 'tensor'
#        # shape of tensor file data is often nx, ny, nz, 1, 6
#        # Get rid of singleton dimension in next-to-last axis
#        if mu1.shape[mask.ndim] == 1:
#            mu1 = mu1[..., 0, :]
#        if mu2.shape[mask.ndim] == 1:
#            mu2 = mu2[..., 0, :]
#    else:
#        raise ValueError("last dimension of mu1 and mu2 should have "
#                         "sizes 2 (colatitute-longitude) or 6 (positive-"
#                         "definite symmetric tensor). Detected %d and %d."
#                         % (mu1.shape[-1], mu2.shape[-1]))

    ROI_size = np.sum(mask > 0)
    frac1 = frac1[mask > 0]
    frac2 = frac2[mask > 0]
    mu1 = mu1[mask > 0]
    mu2 = mu2[mask > 0]

    # Prepare output. ! In a voxel with only one fascicle, that fascicle
    # must be Population zero in the peaks file.
    frac_clean = np.zeros((ROI_size, 2))
    frac_clean[:, 0] = frac1
    frac_clean[:, 1] = frac2
    peaks = np.zeros((ROI_size, 6))
    num_fasc = np.ones(ROI_size) * 2

    if peakmode == 'colat_longit':
        # From colatitude-longitude to x-y-z coordinates
        (x1, y1, z1) = (np.sin(mu1[..., 0]) * np.cos(mu1[..., 1]),
                        np.sin(mu1[..., 0]) * np.sin(mu1[..., 1]),
                        np.cos(mu1[..., 0]))
        (x2, y2, z2) = (np.sin(mu2[..., 0]) * np.cos(mu2[..., 1]),
                        np.sin(mu2[..., 0]) * np.sin(mu2[..., 1]),
                        np.cos(mu2[..., 0]))
        peaks[:, 0] = x1
        peaks[:, 1] = y1
        peaks[:, 2] = z1
        peaks[:, 3] = x2
        peaks[:, 4] = y2
        peaks[:, 5] = z2
    elif peakmode == 'peaks':
        peaks[:, :3] = mu1
        peaks[:, 3:6] = mu2
    elif peakmode == 'tensor':
        # Get eigenvectors (eigenvalues in ascending order)
        (d1, eigv1) = np.linalg.eigh(
            mfu.DT_vec_to_2Darray(mu1)
            )
        (d2, eigv2) = np.linalg.eigh(
            mfu.DT_vec_to_2Darray(mu2)
            )
        # Keep main eigenvector in each voxel. Keep zero vectors
        # for zero matrices (eigh returns matrix of unit
        # eigenvectors instead of all zeros):
        mask_ev1 = (np.abs(d1)[..., -1] > 0)[:, np.newaxis]  # (Nx, Ny, Nz, 1)
        peaks[:, :3] = eigv1[..., -1] * mask_ev1  # (Nx, Ny, Nz, 3)
        mask_ev2 = (np.abs(d2)[..., -1] > 0)[:, np.newaxis]
        peaks[:, 3:6] = eigv2[..., -1] * mask_ev2
    else:
        raise RuntimeError("Unknown peak mode %s (should never get here)."
                           % (peakmode,))

    # Detect and merge confounded directions into direction 1
    dp_max = np.cos(ang_min * np.pi / 180)
    dp = np.sum(peaks[:, :3] * peaks[:, 3:6], axis=-1)  # x1*x2 + y1*y2 + z1*z2
    # dp = x1*x2 + y1*y2 + z1*z2
    dp_abs = np.abs(np.clip(dp, -1, 1))
    merge = dp_abs > dp_max
    n_merge = np.sum(merge)
    if n_merge > 0:
        sign_dp = np.sign(dp[merge])
        # Account for sign of dot product before merging !
        peaks[merge, :3] = (peaks[merge, :3] +
                            peaks[merge, 3:6] * sign_dp[:, np.newaxis])
        norm_merged = np.sqrt(np.sum(peaks[merge, :3]**2, axis=1))
        peaks[merge, :3] = peaks[merge, :3]/norm_merged[:, np.newaxis]
        peaks[merge, 3:6] = 0
        frac_clean[merge, 0] = frac1[merge] + frac2[merge]
        frac_clean[merge, 1] = 0
        # there can be at most 1 fascicle in those voxels now
        num_fasc[merge] = 1

    # Get rid of fascicles relatively too small compared to dominant fascicle
    # Case 1: fascicle 0 too small, transfer fasc 1 over to fasc 0
    f0small = ((frac_clean[:, 1] > ratio * frac_clean[:, 0]) &
               (frac_clean[:, 0] < w_keep))
    if np.sum(f0small) > 0:
        peaks[f0small, :3] = peaks[f0small, 3:6]
        peaks[f0small, 3:6] = 0
        frac_clean[f0small, 0] = frac_clean[f0small, 1]
        frac_clean[f0small, 1] = 0
        num_fasc[f0small] = (frac_clean[f0small, 0] > 0) * 1

    # Case 2: fascicle 1 too small, simply discard it without weight transfer
    f1small = ((frac_clean[:, 0] > ratio * frac_clean[:, 1]) &
               (frac_clean[:, 1] < w_keep))
    if np.sum(f1small) > 0:
        peaks[f1small, 3:6] = 0
        frac_clean[f1small, 1] = 0
        num_fasc[f1small] = (frac_clean[f1small, 0] > 0) * 1

    # Get rid of small (absolute) weights ignored by previous step
    # (one weight could be comparable to another weight but both weights
    # could still be very small)
    w0small = frac_clean[:, 0] < w_small
    if np.sum(w0small) > 0:
        peaks[w0small, :3] = peaks[w0small, 3:6]
        peaks[w0small, 3:6] = 0
        frac_clean[w0small, 0] = frac_clean[w0small, 1]
        frac_clean[w0small, 1] = 0
        num_fasc[w0small] = (frac_clean[w0small, 0] > 0) * 1

    # Easier case where second fascicle has a very low absolute weight
    w1small = frac_clean[:, 1] < w_small
    if np.sum(w1small) > 0:
        peaks[w1small, 3:6] = 0
        frac_clean[w1small, 1] = 0
        num_fasc[w1small] = (frac_clean[w1small, 0] > 0) * 1

    # Return
    peaks_out = np.zeros(mask.shape + (6,))
    peaks_out[mask > 0] = peaks
    num_fasc_out = np.zeros(mask.shape)
    num_fasc_out[mask > 0] = num_fasc
    return peaks_out, num_fasc_out


# top-level definition to make it pickleable for use in parallel estimation
# with Python's multiprocessing module. Called by MFModel.fit()
def _fit_voxel(i, vox_data, sm):
    """Performs fingerprinting in one voxel.

    Returns:
        A vector (1-D NumPy array) containing the voxel parameters of the
        estimated model. The order of the parameters follows the convention
        used in MFModel.fit().
    """
    st_vox = time.time()

    # Unpack shared memory
    ROI_size = sm['ROI_size']
    pgse_scheme = sm['pgse_scheme']
    ms_interpolator = sm['ms_interpolator']
    num_atom = sm['num_atom']
    num_ear = sm['num_ear']
    maxfasc = sm['maxfasc']
    csf_on = sm['csf_on']
    ear_on = sm['ear_on']
    VRB = sm['VRB']
    disp_int = sm['disp_int']
    D = sm['D']  # placeholder for dictionary (saves mem in serial processing)

    # Unpack voxel-specific data
    y = vox_data['y']
    K = vox_data['K']
    csf_i = vox_data['csf_i']
    ear_i = vox_data['ear_i']
    peaks_i = vox_data['peaks']

    num_seq = pgse_scheme.shape[0]
    dicsize = (K*num_atom
               + (csf_i > 0)
               + (ear_i > 0) * num_ear)

    # Parameter order: initial magnetization M0, nu_fasc, ID_fasc, nu_csf,
    # nu_ear, ID_ear, MSE, R2 (coeff. determination)
    i_csf = 2*maxfasc + 1
    i_ear = 2*maxfasc + csf_on + 1
    i_mse = 2*maxfasc + csf_on + 2*ear_on + 1
    i_R2 = 2*maxfasc + csf_on + 2*ear_on + 2
    num_params = 1 + maxfasc*2 + csf_on + 2*ear_on + 2

    # Pre-allocate output
    params_vox = np.zeros(num_params)

    # Skip voxel if no compartment specified
    if K + csf_i + ear_i == 0:
        return params_vox

    # Perform rotations and assemble voxel dictionary
    for k in range(K):
        st = k*num_atom
        end = (k+1)*num_atom
        D[:, st:end] = mfu.interp_PGSE_from_multishell(
            pgse_scheme,
            newdir=peaks_i[3*k:3*k+3],
            msinterp=ms_interpolator)
    subdic_sizes = [num_atom] * K  # Python list here

    # Add optional compartments to dictionary
    if csf_i:
        D[:, K*num_atom] = sm['sig_csf']
        subdic_sizes.append(1)
    if ear_i:
        st = K*num_atom + (csf_i > 0)
        fin = st + num_ear
        D[:, st:fin] = sm['sig_ear']  # only if needed
        subdic_sizes.append(num_ear)

    # Perform combinatorial non-negative linear least squares
    subdic_sizes = np.atleast_1d(subdic_sizes)  # to NumPy array

    (w_nnz,
     ind_subdic,
     ind_totdic,
     SoS,
     y_rec) = mfu.solve_exhaustive_posweights(D[:, :dicsize],
                                              y,
                                              subdic_sizes)
    M0_vox = np.sum(w_nnz)
    nu = w_nnz/M0_vox

    # Return results as a (num_params,) array

    # Baseline signal M0 after factoring out NMR relaxation and diffusion
    # attenuation (representative of initial transverse magnetization after
    # 90-degree radio-frequency pulse):
    params_vox[0] = M0_vox
    # Physical volume fractions of fascicle(s):
    params_vox[1:(K+1)] = nu[:K]  # (voxel-wise K)
    # Index of selected DW-MRI fingerprint in each fascicle sub-dictionary:
    params_vox[(1+maxfasc):(1+maxfasc+K)] = ind_subdic[:K]
    if csf_i:
        # Physical volume fraction of cerebrospinal fluid compartment:
        params_vox[i_csf] = nu[K]
    if ear_i:
        # Physical volume fraction of extra-axonal restricted compartment:
        params_vox[i_ear] = nu[K + (csf_i > 0)]
        # Index of selected DW-MRI fingerprint in EAR sub-dictionary:
        params_vox[i_ear + 1] = ind_subdic[K + (csf_i > 0)]
    # Mean-squared-error in voxel between measured and predicted signal:
    params_vox[i_mse] = SoS/num_seq
    # Coefficient of determination, essentially Pearson's correlation
    # coefficient squared in voxel between measured and predicted signal
    if num_seq > 1 and np.std(y_rec) > 0 and np.std(y) > 0:
        params_vox[i_R2] = np.corrcoef(y, y_rec)[0, 1]**2

    time_vox = time.time() - st_vox

    # Display progress
    if i % disp_int == 0 and VRB >= 3:
        print("Voxel %d/%d (%d fasc%s%s) estimated in %g sec." %
              (i+1, ROI_size, K,
               (", CSF comp" if csf_i else ""),
               (", EAR comp" if ear_i else ""),
               time_vox))
    return params_vox


class MFModel():
    r""" Class for the Microstructure Fingerprinting model.
    """
    MAX_FASC = 2  # max number fascicles in a voxel
    MAX_PROG_LINES = 100  # max number of times fitting progress is displayed
    DFT_DISP_ITVL = 5  # default interval (in voxels) for printing progress

    def __init__(self, dictionary):
        r""" Microstructure Fingerprinting model [1].

        Parameters
        ----------
        dictionary : str or dict
            str must be the path to a Matlab mat file.
            Ask the author for help in generating a dictionary specific to
            your study.

        Notes
        --------
        Currently only implemented for PGSE, HARDI-like acquisition schemes.
        Shells can have different timing (Delta, delta) parameters and echo
        times however.

        References
        ----------
        .. [1] Rensonnet, G., Scherrer, B., Girard, G., Jankovski, A.,
        Warfield, S.K., Macq, B., Thiran, J.P. and Taquet, M., 2019. Towards
        microstructure fingerprinting: Estimation of tissue properties from
        a dictionary of Monte Carlo diffusion MRI simulations. NeuroImage,
        184, pp.964-980.
        """

        if isinstance(dictionary, str):
            self.dic = mfu.loadmat(dictionary)
        elif isinstance(dictionary, dict):
            self.dic = dictionary
        else:
            raise ValueError("Dictionary should either be a valid path to a"
                             " Matlab-like mat file or a Python dictionary.")
        # Compute multi-shell interpolator object
        # TODO: do upon dictionary creation, pickle/dill it and reuse...
        self.ms_interpolator = mfu.init_PGSE_multishell_interp(
            self.dic['dictionary'],
            self.dic['sch_mat'],
            self.dic['orientation'])
        print("Initiated model based on dictionary with %d single-fascicle"
              " fingerprint(s) and %d fingerprint(s) for the extra-axonal"
              " restricted (EAR) compartment." %
              (self.dic['num_atom'], self.dic['num_ear']))
        # TODO: check consistency of dictionary

    # TODO: remove this function definition and use mf_utils
    # get_PGSE_scheme_from_bval_bvec
    def _get_sch_mat_from_bval_bvec(self, bvals, bvecs):
        """Generates PGSE scheme matrix from bval and bvec files or arrays.

        The values of Delta, delta and TE must have been provided to the
        upon instantiation, for instance in a dense, multi-shell PGSE scheme
        matrix.

        Parameters
        ----------
        bvals: str or NumPy array
            b-values in ms/mm^2 (typically around 1000)
        bvecs: str or NumPy array
            unit-norm 3D vectors (will be transposed automatically)

        Returns
        ---------
        sch_mat: 2-D NumPy array with shape (n_seq, 7)
        """
        sch_mat_ref = self.dic['sch_mat']
        if isinstance(bvals, str):
            bvals = np.loadtxt(bvals)
        if isinstance(bvecs, str):
            bvecs = np.atleast_2d(np.loadtxt(bvecs))

        # bvals from s/mm^2 to s/m^2
        bvals = bvals * 1e6

        # Check bvecs
        if np.ndim(bvecs) != 2:
            raise ValueError("bvecs array should have 2 dimensions,"
                             " detected %d." % bvecs.ndim)
        if bvecs.shape[0] != bvals.size and bvecs.shape[1] != bvals.size:
            raise ValueError("Number of b-vectors does not match number"
                             " of b-values (%d)" % bvals.size)

        # Preallocate sheme matrix and transpose bvecs if needed
        sch_mat = np.zeros((bvals.size, 7))
        if bvecs.shape[0] == 3:
            sch_mat[:, :3] = bvecs.transpose()
        elif bvecs.shape[1] == 3:
            sch_mat[:, :3] = bvecs
        else:
            raise ValueError("Vectors in bvecs should be 3-dimensional."
                             " However, detected no dimension with size 3.")

        # Normalize gradient directions
        gnorm = np.sqrt(np.sum(sch_mat[:, :3]**2, axis=1))
        sch_mat[gnorm > 0, :3] = (sch_mat[gnorm > 0, :3] /
                                  gnorm[gnorm > 0][:, np.newaxis])

        # Get gradient intensity from bval assuming unique Delta/deta
        gam = mfu.get_gyromagnetic_ratio('H')
        Del_prot = sch_mat_ref[0, 4]
        del_prot = sch_mat_ref[0, 5]
        TE_prot = sch_mat_ref[0, 6]
        G = np.sqrt(bvals/(Del_prot - del_prot/3))/(gam*del_prot)
        Geff = np.zeros(bvals.shape[0])

        # Map each bval to reference G within a given tolerance
        G_target = np.unique(sch_mat_ref[:, 3])
        Gtol = 1e-3
        G_un_eff = np.zeros(G_target.size)

        grads_per_shell = np.zeros(G_target.size)  # for sanity check
        for ig in range(G_target.size):
            i_shell = np.where(np.abs(G_target[ig] - G) < Gtol)[0]
            grads_per_shell[ig] = i_shell.size
            G_un_eff[ig] = G_target[ig]  # np.mean(G[i_shell])
            Geff[i_shell] = G_target[ig]
        chk = G.size == np.sum(grads_per_shell)
        assert chk, ("%d distinct b-values vs expected %d" %
                     (np.sum(grads_per_shell), G.size))
        sch_mat[:, 3] = Geff

        # Copy and paste unique reference timing parameters
        sch_mat[:, 4:7] = np.array([Del_prot, del_prot, TE_prot])
        return sch_mat

    def fit(self,
            data, mask, numfasc, *,  # named keyword arguments after this
            peaks=None, colat_longit=None, tensors=None,  # requires 1 of 3
            pgse_scheme=None, bvals=None, bvecs=None,
            csf_mask=None, ear_mask=None,  # optional
            verbose=1, parallel=False
            ):
        r""" Perform fingerprinting on pre-computed dictionary of MC signals.

        Parameters
        ----------
        data : str or NumPy array
            str must be the path to a NIfTI file containing an array.
            DWI data for each voxel is assumed to be held in the last
            dimension of the data array, which must have 2 dimensions or more.
        mask : str or NumPy array
            str must be the path to a NIfTI file containing an array.
            A value greater than 0 indicates that the estimation should be
            performed in voxel. Should have shape equal to data.shape[:-1].
        numfasc : str or NumPy array or scalar
            str must be the path to a NIfTI file containing an array.
            Should have shape equal to mask.shape.
            Scalar should be a non-negative integer and will assign its
            value to all data voxels.

        (one of peaks, colat_longit or tensors required)

        peaks : str or NumPy array (optional)
            str must be the path to a NIfTI file containing an array.
            Last dimension of array should be a multiple of 3, where
            peaks[..., 3*i:(3+1)*i] is a unit vector specifying the
            orientation of fascicle i.
        colat_longit : str or NumPy array or list thereof (optional)
            For data with voxels containing multiple fascicles, list required
            with colat_longit[i] a str or NumPy array describing fascicle i.
            str must be the path to a NIfTI file containing an array.
            In earch array, the last dimension should have size 2 and contain
            the angle from the z-axis (theta or colatitute) in [0, pi] and
            the angle from the x-axis in the xy-plane (phi or longitude) in
            [0, 2*pi].
        tensors : str or NumPy array or list thereof (optional)
            For data with voxels containing multiple fascicles, list required
            with tensors[i] a str or NumPy array describing fascicle i.
            str must be the path to a NIfTI file containing an array.

        (either pgse_scheme OR bvals and bvecs required)

        pgse_scheme : str or NumPy array
            str must be the path to a text file with a one-line header.
            Should have shape (Nseq, 7) where Nseq is the number of PGSE
            measurements in each voxel. Each row must be of the form [gx, gy,
            gz, G, Delta, delta, TE] with sqrt(gx**2 + gy**2 + gz**2)=1.
        bvals : str or NumPy array
            str must be the path to a text file
            array should have shape (Nseq,)
            The b-values should have units of ms/mm^2 (~1000), typical of
            clinical usage.
        bvecs : str or NumPy array
            Should contain unit-norm vectors. Shape can be (Nseq, 3) or
            (3, Nseq) interchangeably.

        (always optional)

        csf_mask : str or NumPy array or scalar (optional)
            str must be the path to a NIfTI file containing an array.
            Entries x such that x>0 evaluates to True indicate voxels with a
            cerebrospinal fluid (CSF) compartment.
            scalar assigns its value to all data voxels.
            If not provided, default is no no CSF compartment estimated.
        ear_mask : str or NumPy array or scalar (optional)
            str must be the path to a NIfTI file containing an array.
            Entries x such that x>0 evaluates to True indicate voxels with a
            extra-axonal restricted (EAR) compartment.
            scalar assigns its value to all data voxels.
            If not provided, default is no EAR compartment estimated.

        verbose : 0 (no display), 1 (important info), 2 (detailed info)
            or 3 (all info) printed to standard output

        Returns
        ---------
        An instance of MFModelFit(), with an attribute for each
        estimated microstructural parameter in the form of a NumPy array
        with the same shape as the provided mask.


        Notes
        --------
        Currently only implemented for PGSE, HARDI-like acquisition schemes.
        Shells can have different timing (Delta, delta) parameters and echo
        times however.


        References
        ----------
        .. [1] Rensonnet, G., Scherrer, B., Girard, G., Jankovski, A.,
        Warfield, S.K., Macq, B., Thiran, J.P. and Taquet, M., 2019. Towards
        microstructure fingerprinting: Estimation of tissue properties from
        a dictionary of Monte Carlo diffusion MRI simulations. NeuroImage,
        184, pp.964-980.
        """
        VRB = verbose
        # ------------------
        # Required arguments
        # ------------------
        # DWI Data
        nii_affine = None  # spatial affine transform for DWI data
        if isinstance(data, str):
            st_0 = time.time()
            if VRB >= 2:
                print("Loading data from file %s..." % data)
            nii_affine = nib.load(data).affine
            data_arr = nib.load(data).get_data()
            dur_0 = time.time() - st_0
            if VRB >= 2:
                print("Data loaded in %g s." % dur_0)
        else:
            data_arr = data  # no need to copy, won't be modified

        # ROI mask
        if isinstance(mask, str):
            if nii_affine is None:
                nii_affine = nib.load(mask).affine
            mask_arr = nib.load(mask).get_data()
        else:
            mask_arr = mask  # no need to copy, won't be modified

        img_shape = mask_arr.shape
        ROI = np.where(mask_arr > 0)  # (x,) (x,y) or (x,y,z)
        ROI_size = ROI[0].size

        if ROI_size == 0:
            raise ValueError("No voxel detected in mask. Please provide "
                             "a non-empty mask.")

        if data_arr.shape[:-1] != img_shape:
            raise ValueError("Data and mask not compatible. Based on data,"
                             " mask should have shape (%s), "
                             "got (%s) instead." %
                             (" ".join("%d" % x
                                       for x in data_arr.shape[:-1]),
                              " ".join("%d" % x for x in img_shape)))

        # Number of fascicles in model
        if np.isscalar(numfasc) and not isinstance(numfasc, str):
            # scalar indicator provided for the whole data
            numfasc_roi = np.full(ROI_size, numfasc, dtype=np.int)
        else:  # non scalar mode (array of array in file)
            if isinstance(numfasc, str):
                # Strictly speaking, the array here is not restricted to the
                # ROI yet but it will be later. This way we don't keep a big
                # array in memory after reduction to ROI.
                numfasc_roi = nib.load(numfasc).get_data()
            else:  # NumPy array
                numfasc_roi = numfasc
            nfasc_sh = numfasc_roi.shape
            if mask_arr.shape != nfasc_sh:
                raise ValueError("Data and argument numfasc not compatible. "
                                 " Based on data, numfasc should have "
                                 "shape (%s), got (%s) instead." %
                                 (" ".join("%d" % x for x in img_shape),
                                  " ".join("%d" % x for x in nfasc_sh)))
            # reduce to ROI:
            numfasc_roi = numfasc_roi[mask_arr > 0].astype(np.int)

        maxfasc = int(np.max(numfasc_roi))
        if maxfasc > MFModel.MAX_FASC:
            raise ValueError("Detected %d mask voxel(s) in numfasc with"
                             " number of axon populations greater than"
                             " allowed maximum of %d." %
                             (np.sum(numfasc_roi > MFModel.MAX_FASC),
                              MFModel.MAX_FASC))

        # -------------------
        # (required) Fascicle direction(s)
        #   via one of three options: peaks=, colat_longit=, tensors=
        # -------------------
        peaks_set = False
        if peaks is not None:
            if isinstance(peaks, str):
                # Not strictly in ROI yet, but the name is used to avoid
                # keeping big array in memory after reduction to ROI below
                peaks_roi = nib.load(peaks).get_data()
                if nii_affine is None:
                    nii_affine = nib.load(peaks).affine
            else:  # NumPy array
                peaks_roi = peaks  # same remark as above for "ROI"
            pk_sh = peaks_roi.shape
            if pk_sh[:-1] != img_shape:
                raise ValueError("Arg. peaks not compatible. Based on data,"
                                 " it should have shape (%s x), with x a "
                                 "multiple of 3. Got (%s) instead." %
                                 (" ".join("%d" % x for x in img_shape),
                                  " ".join("%d" % x for x in pk_sh)))
            if pk_sh[-1] % 3 != 0:
                raise ValueError("Size of last dimension of arg. peaks should"
                                 " be a multiple of 3, got %d instead." %
                                 pk_sh[-1])
            if pk_sh[-1] > maxfasc * 3 and VRB >= 1:
                print("Ignoring last %d value(s) along last dimension of"
                      " peaks, as max number of axon populations in mask"
                      " is %d." %
                      (pk_sh[-1] - maxfasc * 3, maxfasc))
            # Internal peaks array has shape (np.sum(mask>0), 3*MAX_FASC)
            peaks_roi = peaks_roi[mask_arr > 0, :3*maxfasc]
            peaks_set = True
        elif colat_longit is not None:
            peak_arg = colat_longit
            datadim = ((2,),)  # colatitute and longitude
        elif tensors is not None:
            peak_arg = tensors
            datadim = ((6,), (1, 6))  # real-valued symmetric tensor
        else:
            raise RuntimeError("At least one of peaks, colat_longit and"
                               " tensors must be specified.")

        # Create peaks array for internal use based on user-provided input,
        # input being either a colatitue/longitude file or list of files, or
        # a tensor file or list of files
        if not peaks_set:
            # Make it a list
            if not isinstance(peak_arg, list):
                peak_arg = [peak_arg]
            peaks_roi = np.zeros((ROI_size, 3 * len(peak_arg)))
            # Iterate through list, ignoring axon populations exceeding
            # the limit (so as to avoid using unnecessary memory)
            if len(peak_arg) > maxfasc and VRB >= 1:
                print("Ignoring %d peak orientation argument(s) because"
                      " max number of axon populations in mask is %d." %
                      (len(peak_arg) - maxfasc, maxfasc))
            for i in range(np.min([len(peak_arg), maxfasc])):
                if isinstance(peak_arg[i], str):
                    peak_arg_i = nib.load(peak_arg[i]).get_data()
                    if nii_affine is None:
                        nii_affine = nib.load(peak_arg[i]).affine
                else:
                    peak_arg_i = peak_arg[i]
                peak_i_sh = peak_arg_i.shape
                if peak_i_sh not in [img_shape + d for d in datadim]:
                    data_dim_str = " or ".join(
                            "(" + " ".join("%d" % x for x in img_shape + ddim)
                            + ")"
                            for ddim in datadim)
                    msg = ("Peak orientation arg. %d of %d seems "
                           "incompatible. Based on data, it should have"
                           " shape %s, got (%s) instead." %
                           (i+1, len(peak_arg),
                            data_dim_str,
                            " ".join("%d" % x for x in peak_i_sh))
                           )
                    raise ValueError(msg)
                if colat_longit is not None:
                    # x-component = sin(th)*cos(phi)
                    peaks_roi[:,
                              3*i + 0] = (np.sin(peak_arg_i[mask_arr > 0, 0]) *
                                          np.cos(peak_arg_i[mask_arr > 0, 1]))
                    # y-component = sin(th)*sin(phi)
                    peaks_roi[:,
                              3*i + 1] = (np.sin(peak_arg_i[mask_arr > 0, 0]) *
                                          np.sin(peak_arg_i[mask_arr > 0, 1]))
                    # z-component = cos(th)
                    peaks_roi[:,
                              3*i + 2] = np.cos(peak_arg_i[mask_arr > 0, 0])
                elif tensors is not None:
                    # Get rid of singleton dimension in next-to-last axis
                    if peak_i_sh[mask_arr.ndim] == 1:
                        idx = ((slice(None),) * mask_arr.ndim
                               + (0,) + (slice(None),))
                        peak_arg_i = peak_arg_i[idx]
                    # Get eigenvectors (eigenvalues in ascending order)
                    (d, eigv) = np.linalg.eigh(
                        mfu.DT_vec_to_2Darray(peak_arg_i[mask_arr > 0, :])
                        )  # shape of tensor file data is nx, ny, nz, 1, 6
                    # Keep main eigenvector in each voxel. Keep zero vectors
                    # for zero matrices (eigh returns matrix of unit
                    # eigenvectors):
                    peaks_roi[:,
                              3*i:3*i+3] = (
                        eigv[..., -1] * (np.abs(d)[..., -1] > 0)[:,
                                                                 np.newaxis])

        for i in range(maxfasc):
            n = i + 1
            peak_L1norm = np.sum(
                np.abs(peaks_roi[numfasc_roi >= n, (n-1)*3:3*n]),
                axis=1)
            num_0 = np.sum(peak_L1norm == 0)
            if num_0 > 0:
                raise ValueError("Detected %d voxel(s) in which the main "
                                 "orientation of axon population %d/%d was "
                                 "a zero vector, although numfasc "
                                 "specifies the presence of that "
                                 "population." %
                                 (num_0, n, maxfasc))

        # -----------------------------
        # (required) Subject-specific protocol information
        #  either schemefile OR bvals and bvecs
        # -----------------------------
        if pgse_scheme is not None:
            # Full PGSE protocol provided in file or in array directly
            if isinstance(pgse_scheme, str):
                # text file provided
                pgse_scheme = np.loadtxt(pgse_scheme, skiprows=1)
            if pgse_scheme.shape[1] != 7:
                raise ValueError("pgse_scheme should have 7 columns, "
                                 " detected %d instead." %
                                 (pgse_scheme.shape[1],))
        else:
            # bval and bvec text provided via text files or arrays
            if bvals is None or bvecs is None:
                raise TypeError("If no schemefile is provided, then both"
                                " bvals and bvecs must be specified.")
            pgse_scheme = self._get_sch_mat_from_bval_bvec(bvals, bvecs)
        num_seq = pgse_scheme.shape[0]
        gam = mfu.get_gyromagnetic_ratio('H')
        G = pgse_scheme[:, 3]
        Delta = pgse_scheme[:, 4]
        delta = pgse_scheme[:, 5]
        TE = pgse_scheme[:, 6]
        b = (gam*G*delta)**2 * (Delta - delta/3)

        # ------------------------
        # Optional model compartments
        # cerebrospinal fluid, extra-axonal restricted
        # ------------------------
        if csf_mask is None:
            csf_mask = np.zeros(ROI_size, dtype=np.bool)
        elif np.isscalar(csf_mask) and not isinstance(csf_mask, str):
            # scalar indicator provided for the whole data
            csf_mask = np.full(ROI_size, csf_mask > 0, dtype=np.bool)
        else:
            # mask covering the whole data volume provided
            if isinstance(csf_mask, str):  # from a file
                csf_mask = nib.load(csf_mask).get_data()
                if nii_affine is None:
                    nii_affine = nib.load(csf_mask).affine
            # At this point, csf_mask must be a NumPy array with ndim>=2
            if csf_mask.shape != img_shape:
                raise ValueError("Arg. csf_mask incomptabible. Based on data,"
                                 " it should have shape (%s), detected (%s)"
                                 " instead." %
                                 (" ".join("%d" % x for x in img_shape),
                                  " ".join("%d" % x for x in csf_mask.shape)))
            # Reduce array to mask size to reduce memory requirements
            csf_mask = csf_mask[mask_arr > 0]
        csf_on = np.any(csf_mask > 0)

        if ear_mask is None:
            ear_mask = np.zeros(ROI_size, dtype=np.bool)
        elif np.isscalar(ear_mask) and not isinstance(ear_mask, str):
            # scalar indicator provided for the whole data
            ear_mask = np.full(ROI_size, ear_mask > 0, dtype=np.bool)
        else:
            # mask covering the whole data volume provided
            if isinstance(ear_mask, str):  # from a file
                ear_mask = nib.load(ear_mask).get_data()
                if nii_affine is None:
                    nii_affine = nib.load(ear_mask).affine
            # At this point, ear_mask must be a NumPy array with ndim>=2
            if ear_mask.shape != img_shape:
                raise ValueError("Arg. ear_mask incomptabible. Based on data,"
                                 " it should have shape (%s), detected (%s)"
                                 " instead." %
                                 (" ".join("%d" % x for x in img_shape),
                                  " ".join("%d" % x for x in ear_mask.shape)))
            # Reduce array to mask size to reduce memory requirements
            ear_mask = ear_mask[mask_arr > 0]
        ear_on = np.any(ear_mask > 0)

        n_empty = np.sum((numfasc_roi + csf_mask + ear_mask) == 0)
        if n_empty > 0 and VRB >= 2:
            print("WARNING: detected %d voxel(s) in mask with zero "
                  " axon population, no cerebrospinal fluid (CSF) and no"
                  " extra-axonal restricted (EAR) compartment specified."
                  " No estimation will be performed there." % (n_empty,))

        # ------------------------
        # Optional: request parallel processing
        # ------------------------
        if parallel and mfu.from_ipython() and os.name == 'nt':
            msg = ("Parallel mode cannot be run from IPython environement"
                   " (e.g., Spyder GUI) on Windows due to "
                   "incompatibilities with "
                   "Python's multiprocessing module. Run the code by"
                   " directly calling the python interpreter from a "
                   "regular console.")
            raise RuntimeError(msg)

        # --------------------------------
        # Patient-specific CSF and extra-axonal restricted dictionaries
        # --------------------------------
        if csf_on:
            sig_csf = (np.exp(-TE/self.dic['T2_csf']) *
                       np.exp(-b*self.dic['DIFF_csf']))
        if ear_on:
            sig_ear = np.zeros((num_seq, self.dic['num_ear']))
            for i in range(self.dic['num_ear']):
                sig_ear[:, i] = (np.exp(-TE/self.dic['T2_ear']) *
                                 np.exp(-b*self.dic['DIFF_ear'][i]))

        # --------------------------------
        # Pre-allocate space for multi-compartment dictionary
        # -------------------------------
        num_atom = self.dic['num_atom']
        max_dicsize = (maxfasc * num_atom +
                       csf_on +
                       ear_on * self.dic['num_ear'])
        D = np.zeros((num_seq, max_dicsize))

        # Note: peaks, numpeaks, csf and ear are reduced to ROI shape
        # (np.sum(mask>0),)

        # Parameter order:
        # initial magnetization M0,
        # nu_fasc, ID_fasc,
        # nu_csf,
        # nu_ear, ID_ear,
        # MSE, R2 (coeff. determination)
        num_params = 1 + maxfasc*2 + csf_on*1 + ear_on*2 + 2

        # Display interval (for printing progress every x voxels)
        disp_int = int(ROI_size/np.min([ROI_size/MFModel.DFT_DISP_ITVL,
                                        MFModel.MAX_PROG_LINES]))

        # -------------------------------
        # Shared memory for external estimation process(es), i.e.
        # info valid for all voxels
        # -------------------------------
        sm = {'D': D,
              'ROI_size': ROI_size,
              'pgse_scheme': pgse_scheme,
              'ms_interpolator': self.ms_interpolator,
              'num_atom': num_atom,
              'num_ear': self.dic['num_ear'],
              'maxfasc': maxfasc,
              'csf_on': csf_on,
              'ear_on': ear_on,
              'VRB': VRB,
              'disp_int': disp_int
              }
        if csf_on:
            sm['sig_csf'] = sig_csf
        if ear_on:
            sm['sig_ear'] = sig_ear

        # -------------------------------
        # Start estimation
        # -------------------------------

        st_est = time.time()

        if parallel:
            # Parallel, multi-process execution
            n_cpu = int(mp.cpu_count()/1)
            # chunksize heuristics: number of physical CPUs usually half of
            # value returned by cpu_count, hence the factor 2.
            chunksize = max(1, int(2*ROI_size/n_cpu))
            if VRB >= 2:
                print("Starting estimation in %d voxel(s) in parallel"
                      " mode, displaying progress every %d voxel(s)." %
                      (ROI_size, disp_int))
            with mp.Pool(n_cpu) as pool:
                # Use iterators to save memory while passing data to processes
                # Position of ROI voxel i:
                pos_iter = (tuple(axis[i] for axis in ROI)
                            for i in range(ROI_size))
                # DWI data in ROI voxel i:
                dwi_iter = (data_arr[next(pos_iter) + (slice(None),)]
                            for _ in range(ROI_size))
                # Voxel-specific info for estimation bundled in Python dict:
                vox_data_iter = ({'csf_i': csf_mask[i],
                                  'ear_i': ear_mask[i],
                                  'K': numfasc_roi[i],
                                  'peaks': peaks_roi[i, :],
                                  'y': next(dwi_iter)}
                                 for i in range(ROI_size))
                args_iter = ((i, next(vox_data_iter), sm)
                             for i in range(ROI_size))
                # Multiprocessing - launch parallel processes
                params_in_mask = pool.starmap_async(_fit_voxel,
                                                    args_iter,
                                                    chunksize).get()
            params_in_mask = np.array(params_in_mask)

        else:
            # single-thread execution
            if VRB >= 2:
                print("Starting estimation in %d voxel(s) in serial "
                      "mode, displaying progress every %d voxel(s)." %
                      (ROI_size, disp_int))
            # Pre-allocate output
            params_in_mask = np.zeros((ROI_size, num_params))
            for i in range(ROI_size):
                vox = [axis[i] for axis in ROI]  # can be 1D, 2D, 3D, ...
                vox = tuple(vox)  # required for array indexing
                y = data_arr[vox + (slice(None),)]  # last dim holds DWI data
                vox_data = {'csf_i': csf_mask[i],
                            'ear_i': ear_mask[i],
                            'K': numfasc_roi[i],
                            'peaks': peaks_roi[i, :],
                            'y': y}
                params_in_mask[i, :] = _fit_voxel(i, vox_data, sm)

        time_est = time.time() - st_est
        if VRB >= 2:
            print("Estimation performed in %g second(s)." % time_est)

        # Return a Dipy-style "fit object" with the info to output model
        # parameters
        fitinfo = {'maxfasc': maxfasc,
                   'csf_on': csf_on,
                   'ear_on': ear_on,
                   'affine': nii_affine,
                   'mask': mask_arr,
                   'fasc_propnames': [x.strip() for x in
                                      self.dic['fasc_propnames']]}
        for n in fitinfo['fasc_propnames']:
            fitinfo['_' + n] = self.dic[n]  # prepend _ for name collisions
        if ear_on:
            fitinfo['DIFF_ear'] = self.dic['DIFF_ear']
        return MFModelFit(fitinfo, params_in_mask, verbose=VRB)


class MFModelFit():
    def __init__(self, fitinfo, model_params, verbose=0):
        """
        """
        self.affine = fitinfo['affine']

        numfasc = fitinfo['maxfasc']
        csf_on = fitinfo['csf_on']
        ear_on = fitinfo['ear_on']
        mask = fitinfo['mask']

        # M0
        self.M0 = np.zeros(mask.shape)
        self.M0[mask > 0] = model_params[:, 0]
        parlist = ['M0']

        # Total fvf, fascicle-specific properties
        fvf_in_mask = 0
        for k in range(numfasc):
            nu_k = model_params[:, k+1]
            prop_map = np.zeros(mask.shape)
            prop_map[mask > 0] = nu_k
            par_name = 'frac_f%d' % k
            setattr(self, par_name, prop_map)
            parlist.append(par_name)

            ID_k = model_params[:, 1+numfasc+k].astype(np.int)
            fvf_k = fitinfo['_fvf'][ID_k] * (nu_k > 0)
            fvf_in_mask += nu_k * fvf_k
            for n in fitinfo['fasc_propnames']:
                # Leave property to zero if no weight assigned to fascicle!
                prop_k_in_mask = fitinfo['_' + n][ID_k] * (nu_k > 0)
                prop_map = np.zeros(mask.shape)
                prop_map[mask > 0] = prop_k_in_mask
                par_name = n + '_f%d' % k
                setattr(self, par_name, prop_map)
                parlist.append(par_name)
        self.fvf_tot = np.zeros(mask.shape)
        self.fvf_tot[mask > 0] = fvf_in_mask
        parlist.append('fvf_tot')

        if csf_on:
            self.frac_csf = np.zeros(mask.shape)
            self.frac_csf[mask > 0] = model_params[:, 2*numfasc + 1]
            parlist.append('frac_csf')

        if ear_on:
            self.frac_ear = np.zeros(mask.shape)
            nu_ear_mask = model_params[:, 2*numfasc + csf_on + 1]
            self.frac_ear[mask > 0] = nu_ear_mask
            parlist.append('frac_ear')

            ID_ear = model_params[:, 2*numfasc + csf_on + 2].astype(np.int)
            self.D_ear = np.zeros(mask.shape)
            self.D_ear[mask > 0] = (fitinfo['DIFF_ear'][ID_ear]
                                    * (nu_ear_mask > 0))
            # leave D_ear to zero if no weight assigned to compartment!
            parlist.append('D_ear')

        # Mean squared error
        self.MSE = np.zeros(mask.shape)
        self.MSE[mask > 0] = model_params[:, -2]
        parlist.append('MSE')

        # R squared (oefficient of determination)
        self.R2 = np.zeros(mask.shape)
        self.R2[mask > 0] = model_params[:, -1]
        parlist.append('R2')

        # Store parameter names
        self.param_names = parlist

        # Display progress and user instructions
        if verbose >= 2:
            print("Microstructure Fingerprinting fit object constructed.")
            print("Assuming the fit object was named \'MF_fit\', "
                  "you can access property maps (NumPy arrays) "
                  "via \'MF_fit.property_name\',"
                  " where \'property_name\' can be any of the following:")
            for p in parlist:
                print('\t%s' % (p,))
            print("You can call \'MF_fit.write_nifti\' to write the "
                  "corresponding NIfTI files.")

    def write_nifti(self, output_basename, affine=None):
        """Exports maps of fitted parameter as NIfTI files.

        Parameters
        ----------
        output_basename: str
            unix/like/path/to/output_file. In order to
            force the creation of compressed .nii.gz archives, provide a
            base name with the .nii.gz extension.
        affine: NumPy array
            Array with shape (4, 4), usually obtained as
            `affine = loaded_nifti_object.affine`. If not specified, an
            attempt will be made at finding an affine transform
            from the NIfTI files provided during the fitting.

        Returns
        -------
        fnames : list of all the files created.
        """
        if affine is None:
            affine = self.affine
        if affine is None:
            # no affine ever given to Fit object
            msg = ("Argument affine must be explicitely passed  because "
                   "no affine transform matrix was found during model "
                   "fitting. Expecting NumPy array with shape (4, 4).")
            raise ValueError(msg)

        # Special case for tarred archives .nii.gz
        niigz = '.nii.gz'
        if (len(output_basename) > len(niigz) and
                output_basename[-len(niigz):] == niigz):
            # case of .nii.gz file extension
            (path, fname) = os.path.split(output_basename[:-len(niigz)])
            ext = niigz
        else:
            # tail never contains a slash
            (path, tail) = os.path.split(output_basename)
            # ext is empty or starts with a period
            (fname, ext) = os.path.splitext(tail)
            if ext not in ['', '.nii']:
                raise ValueError("Unknown NIfTI extension %s in output %s" %
                                 (ext, output_basename))
            ext = '.nii'

        basename = os.path.join(path, fname)
        fnames = []
        for p in self.param_names:
            nii = nib.Nifti1Image(getattr(self, p), affine)
            nii_fname = '%s_%s%s' % (basename, p, ext)
            nib.save(nii, nii_fname)
            fnames.append(nii_fname)
        return fnames
