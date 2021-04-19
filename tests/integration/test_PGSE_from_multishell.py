# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:22:50 2019

For use with pytest. For speed tests, run manually in IPython console.

Test interpolation from pre-initialized functions for every HARDI shell
of protocol. Often used in rotation of single-fascicle signals.

Subject 1000260: true signal from phases vs signal obtained from within-shell
interpolation using the dense sampling:
    with 90-direction sampling: max(max(AbsErr)) = 6.334e-3
    with 400-direction sampling: max(max(AbsErr)) = 3.864e-3
Subject 1000521:
    with 90-direction sampling: max(max(AbsErr)) = 6.0945e-3
    with 400-direction sampling: max(max(AbsErr)) = 6.191e-3
A factor 4 translates in an approximately sqrt(4)-fold decrease in the error.


@author: rensonnetg
"""
import os
import matplotlib
# if os.name == 'posix' and "DISPLAY" not in os.environ:
matplotlib.use('Agg')  # for CI/CD without display
import matplotlib.pyplot as plt
import numpy as np
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

drivedir = os.path.join('E:', os.path.sep, 'rensonnetg', 'OneDrive - UCL')

# %% Check with cat spinal cord data
plt.close('all')
Nreps = 15
gam = mfu.get_gyromagnetic_ratio('hydrogen')  # 2*np.pi*42.577480e6

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fixtures',
    )
ld_dic_ms = mfu.loadmat(os.path.join(FIXTURE_DIR,
                                     'MC_dictionary_cat_spine_3D.mat'))
sch_mat_ms = mfu.import_PGSE_scheme(os.path.join(FIXTURE_DIR,
                                    '3D_qspace_clean_smooth.scheme'))
sch_mat = mfu.import_PGSE_scheme(os.path.join(FIXTURE_DIR,
                                 '3D_qspace_clean.scheme')
                                 )  # G values change even within a shell
sch_mat_smooth = np.loadtxt(os.path.join(FIXTURE_DIR,
                                         '3D_qspace_clean_smooth.scheme'),
                            skiprows=1)  # just for plotting

bvals_ms = (gam * sch_mat_ms[:, 3] *
            sch_mat_ms[:, 5])**2 * (sch_mat_ms[:, 4] - sch_mat[:, 5] / 3)
bvals_new = (gam * sch_mat[:, 3] *
             sch_mat[:, 5])**2 * (sch_mat[:, 4] - sch_mat[:, 5]/3)

# Weird bug: TEs were very slightly different between HARDI shells, making for
# small differences in the b0, unweighted signals
b0_ms = np.where(sch_mat[:, 3] == 0)[0]
sig_ms = ld_dic_ms['dic_fascicle']
# force equality because interpolation expects identical b0 signals (since
# it also expects identical Delta, delta, TE)
sig_ms[b0_ms, :] = sig_ms[b0_ms[0], :]
# make sure Delta, delta, TE identical in scheme matrices
sch_mat[:, 4:7] = sch_mat[0, 4:7]
sch_mat_ms[:, 4:7] = sch_mat[0, 4:7]

# Make sure gradients of new prot included in [min(G), max(G)] of dense scheme
# because extrapolation not supported. A very small correction needs to be
# applied to 0.60001 gradients in new protocol (down to 0.600004)
G_max_interp = np.max(sch_mat_ms[:, 3])
sch_mat[:, 3] = np.minimum(sch_mat[:, 3], G_max_interp)

# Subsample sch_mat, remove some shells, duplicate them, etc.
#seq_new = sch_mat_ms[:, 3] < 0.400
#seq_to_change = (sch_mat[:, 3] > 0.250) & (sch_mat_ms[:, 3] < 0.500)
#sch_mat = sch_mat_ms.copy()
#sch_mat[:, 4:7] = sch_mat_ms[0, 4:7]
#sch_mat[seq_to_change, 3] = sch_mat[seq_to_change, 3] + 0.015
#sch_mat = sch_mat[seq_to_change, :]

ordir = np.array([0, 0, 1])  # make value default
# mean DTI: np.array([0.109565042640809, 0.052956537589947, 0.992567935487341])
newdir = ordir

start_slow = time.time()
for i in range(Nreps):
    sig_new = mfu.interp_PGSE_from_multishell(sch_mat, newdir,
                                              sig_ms, sch_mat_ms, ordir)
time_slow = time.time() - start_slow

start_init = time.time()
for i in range(Nreps):
    ms_interp = mfu.init_PGSE_multishell_interp(sig_ms, sch_mat_ms, ordir)
time_init = time.time() - start_init

start_fast = time.time()
for i in range(Nreps):
    sig_new_fast = mfu.interp_PGSE_from_multishell(sch_mat, newdir,
                                                   sig_ms, sch_mat_ms, ordir,
                                                   ms_interp)
time_fast = time.time() - start_fast


def test_interp_initialized_rodent():
    # Fast and slow signals should be very very close (up to numerical
    # round-off errors)
    chk = np.all(np.abs(sig_new - sig_new_fast) <= 1e-7)
    assert chk, ('Ex vivo rodent 3D protocol, interpolation from dense '
                 'multishell without pre-initialized interpolator (slow) '
                 'seems to differ from interpolation with pre-initialized '
                 'interpolators (fast).')


print("Ex vivo rodent data: time slow %g, time init %g, "
      "time fast %g /run [s]" %
      (time_slow/Nreps, time_init/Nreps, time_fast/Nreps))


# Using smooth scheme matrix for new protocol avoids warning by plotting
# function, which considers different G values as belonging to different
# shells
mfu.plot_multi_shell_signal(sig_ms[:, ::200], sch_mat_ms, ordir)
mfu.plot_multi_shell_signal(sig_new[:, ::200], sch_mat_smooth, newdir)

# %% Check with UKBB data
ukbb_subj = 1000521
bvals_file = os.path.join(FIXTURE_DIR, "%d_bvals.txt" % ukbb_subj)
bvecs_file = os.path.join(FIXTURE_DIR, "%d_bvecs.txt" % ukbb_subj)
bvals = np.loadtxt(bvals_file)
bvecs = np.loadtxt(bvecs_file)


def test_subj_bval_bvec_files():
    # Create subject-specific scheme matrix. INPUTS: bvals, bvecs files.
    chk = bvals.size == bvecs.shape[1]
    assert chk, ("UKBB data number of bvalues (%d) does not match number of"
                 " b-vectors (%d)." % (bvals.size, bvecs.shape[1]))


# Get general dictionary and protocol for whole UKBB study, i.e.
# dense scheme matrix (dense multishell sampling), dense dictionary and
# effective bval based on population averages
gsampling = 90
dense_sch_mat = mfu.import_PGSE_scheme(os.path.join(FIXTURE_DIR,
                                       "ukbb_scheme_%d_"
                                       "dirs.scheme" % gsampling))
# General dictionary for UKBB study (dense multi-shell sampling)
ukbb_dicinfo = mfu.loadmat(os.path.join(FIXTURE_DIR,
                                        'ukbb_%d_dirs_dictionary'
                                        '_hcp_deltas.mat' % gsampling))
# b_eff_ukbb = [5, 999.1, 1997.5]  # based on population averages
b_eff_ukbb = np.unique(ukbb_dicinfo['bvals'][ukbb_dicinfo['bvals'] > 0])
b_eff_ukbb = b_eff_ukbb/1e6  # from s/m^2 to s/mm^2
# G_eff_ukbb = [0.003964, 0.056037, 0.079234]
G_eff_ukbb = np.unique(ukbb_dicinfo['sch_mat'][:, 3])
G_eff_ukbb = G_eff_ukbb[G_eff_ukbb > 0]
btol = 25  # [s/mm^2]

# Get subject-specific dictionary computed from Monte Carlo simulations
# (from final accumulated phases), serving as reference
dic_subj_gt = mfu.loadmat(os.path.join(FIXTURE_DIR,
                                       '%d_dictionary_hcp_'
                                       'deltas.mat' % ukbb_subj))
sch_mat_subj = np.zeros((bvals.size, 7))
sch_mat_subj[:, :3] = np.transpose(bvecs)  # copy subject b-vectors
sch_mat_subj[:, 4:7] = dense_sch_mat[0, 4:7]  # copy Delta, delta, TE
Del_subj = sch_mat_subj[:, 4]
del_subj = sch_mat_subj[:, 5]
# Compute gradient intensities in SI units
sch_mat_subj[:, 3] = (np.sqrt(bvals*1e6/(Del_subj-del_subj/3))
                      / (gam*del_subj))

# Sanity check: check that every subject-specific gradient intensity is close
# enough to reference gradients in dense multishell sampling
grads_per_shell = np.zeros(len(b_eff_ukbb))  # just for sanity check
for ib in range(len(b_eff_ukbb)):
    i_shell = np.where(np.abs(b_eff_ukbb[ib] - bvals) < btol)[0]
    grads_per_shell[ib] = i_shell.size
    # sch_mat_subj[i_shell, 3] = G_eff_ukbb[ib]  # update G column


def test_subj_bvals_close_to_dense_sampling():
    chk = bvals.size == np.sum(grads_per_shell)
    assert chk, ('UKBB subj %d: %d b-values close enough to dense sampling'
                 ' vs expected %d' %
                 (ukbb_subj, np.sum(grads_per_shell), bvals.size))


# avoid extrapolation: reduce max gradient of subject
max_G_dens = np.max(dense_sch_mat[:, 3])
sch_mat_subj[:, 3] = np.minimum(sch_mat_subj[:, 3], max_G_dens)

# Compute subject-specific dictionary
start_slow = time.time()
for i in range(Nreps):
    sig_subj = mfu.interp_PGSE_from_multishell(sch_mat_subj,
                                               ukbb_dicinfo['orientation'],
                                               ukbb_dicinfo['dictionary'],
                                               ukbb_dicinfo['sch_mat'],
                                               ukbb_dicinfo['orientation']
                                               )
time_slow = time.time() - start_slow

# With initialization
start_init
for i in range(Nreps):
    ms_interpolator = (
            mfu.init_PGSE_multishell_interp(ukbb_dicinfo['dictionary'],
                                            ukbb_dicinfo['sch_mat'],
                                            ukbb_dicinfo['orientation']))
time_init = time.time() - start_init

start_fast = time.time()
for i in range(Nreps):
    sig_subj_fast = mfu.interp_PGSE_from_multishell(
            sch_mat_subj,
            ukbb_dicinfo['orientation'],
            msinterp=ms_interpolator)
time_fast = time.time() - start_fast

# AE_fast should be zero everywhere
AE_fast = np.abs(sig_subj - sig_subj_fast)


def test_interp_initialized_UKBB():
    chk_ukbb = np.all(AE_fast < 1e-7)
    assert chk_ukbb, ('UKBB data, interpolation from dense '
                      'multishell without pre-initialized interpolator (slow) '
                      'seems to differ from interpolation with pre-'
                      'initialized interpolators (fast).')


print("UKBB data: time slow %g, time init %g, time fast %g /run [s]" %
      (time_slow/Nreps, time_init/Nreps, time_fast/Nreps))


# Compare with groundtruth computed from MC phases (only for 2 subjects), for
# all microstructure configurations in dictionnary, i.e., all fingerprints

# AE captures the error due to the interpolation instead of computing the
# actual Monte Carlo signal for the rotated fascicles of axons. Subject to the
# variance of Monte Carlo, typically of the order of 0.5e-3 with the simulation
# parameters that we use
AE = np.abs(sig_subj - dic_subj_gt['dictionary'])
print("UKBB data: comparison with groundtruth signals from Monte "
      "Carlo phases, max absolute error %g." % np.max(AE))


def test_interp_from_dense_vs_monte_carlo():
    # Gross check (should be better than this)
    chk = np.all(AE < 1e-2)
    assert chk, ('UKBB data: seemingly big differences (up to %g) between'
                 ' groudtruth Monte Carlo signal and interpolation '
                 'from dense multi-shell scheme.' % (np.max(AE),))
