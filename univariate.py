from functools import partial
from os import environ
from pathlib import Path
from subprocess import PIPE, run

import nibabel as nib
import numpy as np
import pandas as pd
from bids import BIDSLayout
from joblib import Parallel, delayed
from juliacall import Main as jl
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)
from nilearn.image import binarize_img, load_img, mean_img
from nilearn.reporting import get_clusters_table
from scipy.stats import norm


def main():
    """Main function for the full first- and second-/third-level analysis."""

    # Input parameters: File paths
    bids_dir = Path('/ptmp/aenge/slang/data')
    derivatives_dir = bids_dir / 'derivatives'
    fmriprep_dir = derivatives_dir / 'fmriprep'
    output_dir = derivatives_dir / 'univariate'

    # Input parameters: Inclusion/exclusiong criteria
    fd_thresh = 0.5
    df_query = 'subject.str.startswith("SA") & perc_outliers <= 0.25 & n_sessions >= 2'

    # Inpupt parameters: First-level GLM
    task = 'language'
    space = 'MNI152NLin2009cAsym'
    smoothing_fwhm = 5.0
    hrf_model = 'glover + derivative + dispersion'
    contrasts = {
        'audios-noise': (('audios_noise',), ()),
        'audios-pseudo': (('audios_pseudo',), ()),
        'audios-pseudo-minus-noise': (('audios_pseudo',), ('audios_noise',)),
        'audios-words': (('audios_words',), ()),
        'audios-words-minus-noise': (('audios_words',), ('audios_noise',)),
        'audios-words-minus-pseudo': (('audios_words',), ('audios_pseudo',)),
        'images-minus-audios': (('images_noise', 'images_pseudo', 'images_words'),
                                ('audios_noise', 'audios_pseudo', 'audios_words')),
        'images-noise': (('images_noise',), ()),
        'images-pseudo': (('images_pseudo',), ()),
        'images-pseudo-minus-noise': (('images_pseudo',), ('images_noise',)),
        'images-words': (('images_words',), ()),
        'images-words-minus-noise': (('images_words',), ('images_noise',)),
        'images-words-minus-pseudo': (('images_words',), ('images_pseudo',))}

    # Input parameters: Cluster correction
    clustsim_sidedness = '2-sided'
    clustsim_nn = 'NN1'  # Must be 'NN1' for Nilearn's `get_clusters_table`
    clustsim_voxel_thresh = 0.001
    clustsim_cluster_thresh = 0.05
    clustsim_iter = 10000

    # Input parameters: Performance
    # n_jobs = int(environ['SLURM_CPUS_PER_TASK'])
    n_jobs = 8  # When running outside of SLURM

    # Load BIDS structure
    pybids_dir = output_dir / 'pybids'
    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir,
                        database_path=pybids_dir)

    # Fit first-level GLM, separately for each subject and session
    glms, mask_imgs, percs_outliers = run_glms(bids_dir, fmriprep_dir,
                                               pybids_dir, task, space,
                                               fd_thresh, hrf_model,
                                               smoothing_fwhm, n_jobs)

    # Load metadata (subjects, sessions, time points) for mixed model
    df = load_df(layout, task, percs_outliers, df_query)
    good_ixs = list(df.index)

    # Combine brain masks
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]
    mask_img, mask_file = combine_save_mask_imgs(mask_imgs, output_dir,
                                                 task, space)
    mask = mask_img.get_fdata().astype(np.int32)

    # Loop over contrasts
    for contrast_label, (conditions_plus, conditions_minus) in contrasts.items():

        # Compute beta images for each subject and session
        beta_imgs = [compute_beta_img(glm, conditions_plus, conditions_minus)
                     for ix, glm in enumerate(glms) if ix in good_ixs]
        betas = [load_img(img).get_fdata() for img in beta_imgs]
        betas = np.array(betas).squeeze()

        # Save beta images
        subjects = df['subject']
        sessions = df['session']
        for beta_img, subject, session in zip(beta_imgs, subjects, sessions):
            save_beta_img(beta_img, output_dir, subject, session, task,
                          space, contrast_label)

        # Fit linear group-level linear mixed models using Julia
        print(f'Fitting mixed models for contrast "{contrast_label}"...')
        formula = 'beta ~ time + time2 + (time + time2 | subject)'
        voxel_ixs = np.transpose(mask.nonzero())
        betas_per_voxel = [betas[:, x, y, z] for x, y, z in voxel_ixs]
        # # For quick testing: Necessary when using only 1,000 voxels
        # betas_per_voxel = betas_per_voxel[0:1000]
        dfs = [df.assign(beta=betas) for betas in betas_per_voxel]
        res = fit_mixed_models(formula, dfs)
        bs, zs, residuals = zip(*res)
        b0, b1, b2 = np.array(bs).T
        z0, z1, z2 = np.array(zs).T
        residuals = np.array(residuals)

        # # For quick testing: Necessary when using only 1,000 voxels
        # n_pad = mask.sum() - residuals.shape[0]
        # residuals = np.pad(residuals, ((0, n_pad), (0, 0)))

        # Save model residuals to NIfTI file
        n_residuals = residuals.shape[1]
        residuals_ref = np.repeat(mask[:, :, :, np.newaxis],
                                  n_residuals, axis=3)
        residuals_ref_img = nib.Nifti1Image(residuals_ref, mask_img.affine)
        residuals_img, residuals_file = \
            save_array_to_nifti(residuals, residuals_ref_img, voxel_ixs,
                                output_dir, task, space, contrast_label,
                                suffix='residuals')

        # Compute parametric cluster FWE correction threshold using AFNI
        print('Computing cluster correction for ' +
              f'contrast "{contrast_label}"...')
        acf = compute_acf(residuals_file, mask_file)
        cluster_threshold = \
            compute_cluster_threshold(acf, mask_file, clustsim_sidedness,
                                      clustsim_nn, clustsim_voxel_thresh,
                                      clustsim_cluster_thresh, clustsim_iter)
        print(f'Cluster threshold: {cluster_threshold:.1f} voxels')

        # Save unthresholded and thresholded statistical images to NIfTI files
        for array, suffix in zip([b0, b1, b2, z0, z1, z2],
                                 ['b0', 'b1', 'b2', 'z0', 'z1', 'z2']):

            # # For quick testing: Necessary when using only 1,000 voxels
            # n_pad = mask.sum() - array.shape[0]
            # array = np.pad(array, ((0, n_pad)))

            img, file = save_array_to_nifti(array, mask_img, voxel_ixs,
                                            output_dir, task, space,
                                            contrast_label, suffix)

            if suffix.startswith('z'):
                save_clusters(img, clustsim_voxel_thresh, cluster_threshold,
                              output_dir, task, space, contrast_label, suffix)


def run_glms(bids_dir, fmriprep_dir, pybids_dir, task, space, fd_thresh,
             hrf_model, smoothing_fwhm, n_jobs):
    """Runs first-level GLM for all subjects and sessions."""

    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir,
                        database_path=pybids_dir)

    subjects_sessions = get_subjects_sessions(layout, task, space)

    run_glm_ = partial(run_glm, bids_dir, fmriprep_dir, pybids_dir, task,
                       space, fd_thresh, hrf_model, smoothing_fwhm)

    res = Parallel(n_jobs)(delayed(run_glm_)(subject, session)
                           for subject, session in subjects_sessions)

    return zip(*res)


def get_subjects_sessions(layout, task, space):
    """Gets a list of all subject-session pairs with preprocessed data."""

    subjects = layout.get_subjects(task=task, space=space, desc='preproc',
                                   suffix='bold', extension='nii.gz')

    all_sessions = [layout.get_sessions(subject=subject, task=task,
                                        space=space, desc='preproc',
                                        suffix='bold', extension='nii.gz')
                    for subject in subjects]

    subjects_sessions = [(subject, session)
                         for subject, sessions in zip(subjects, all_sessions)
                         for session in sessions]

    return sorted(subjects_sessions)


def run_glm(bids_dir, fmriprep_dir, pybids_dir, task, space,
            fd_thresh, hrf_model, smoothing_fwhm, subject, session):
    """Runs a first-level GLM for a given subject and session."""

    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir,
                        database_path=pybids_dir)

    mask_img = load_mask_img(layout, subject, session, task, space)

    events = load_events(layout, subject, session, task)

    func_img = load_func_img(layout, subject, session, task, space)

    frame_times = make_frame_times(layout, func_img)

    confounds, perc_outliers = get_confounds(layout, subject, session, task,
                                             fd_thresh)

    design_matrix = make_first_level_design_matrix(frame_times,
                                                   events=events,
                                                   hrf_model=hrf_model,
                                                   drift_model=None,
                                                   high_pass=None,
                                                   add_regs=confounds)

    glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=mask_img)
    glm.fit(run_imgs=func_img, design_matrices=[design_matrix])

    return glm, mask_img, perc_outliers


def load_mask_img(layout, subject, session, task, space):
    """Loads the preprocessed brain mask for a given subject and session."""

    mask_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='brain', suffix='mask',
                            extension='nii.gz')
    assert len(mask_files) == 1
    mask_file = mask_files[0]

    return load_img(mask_file)


def load_events(layout, subject, session, task):
    """Loads task events for a given subject and session as a DataFrame."""

    events_files = layout.get(subject=subject, session=session, task=task,
                              suffix='events', extension='tsv')
    assert len(events_files) == 1
    events_file = events_files[0]

    return pd.read_csv(events_file, sep='\t')


def load_func_img(layout, subject, session, task, space):
    """Loads preprocessed fMRI data for a given subject and session."""

    func_files = layout.get(subject=subject, session=session, task=task,
                            space=space, desc='preproc', suffix='bold',
                            extension='nii.gz')
    assert len(func_files) == 1
    func_file = func_files[0]

    return load_img(func_file)


def make_frame_times(layout, func_img):
    """Creates a list of frame times for a given fMRI image."""

    n_scans = func_img.shape[-1]
    tr = layout.get_tr()

    return tr * (np.arange(n_scans) + 0.5)


def get_confounds(layout, subject, session, task, fd_thresh=0.5):
    """Loads a common set of confounds for a given subject and session.

    These are:
    * Six head motion parameters (translations and rotations)
    * Six top aCompCor components
    * Cosine regressors for high-pass filtering
    * Spike regressors for non-steady-state volumes at the beginning of scans
    * Spike regressors for outlier volumes based on framewise displacement
    """

    confounds_files = layout.get(subject=subject, session=session, task=task,
                                 desc='confounds', suffix='timeseries',
                                 extension='tsv')
    assert len(confounds_files) == 1
    confounds_file = confounds_files[0]
    confounds = pd.read_csv(confounds_file, sep='\t')

    hmp_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    compcor_cols = ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
                    'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05']

    cosine_cols = [col for col in confounds if col.startswith('cosine')]

    non_steady_cols = [col for col in confounds
                       if col.startswith('non_steady_state_outlier')]

    confounds, outlier_cols = add_outlier_regressors(confounds, fd_thresh)

    n_outliers = len(outlier_cols)
    n_volumes = len(confounds)
    perc_outliers = n_outliers / n_volumes
    print(f'Found {n_outliers} outliers ({perc_outliers * 100:.1f}%) for ' +
          f'subject {subject}, session {session}')

    cols = hmp_cols + compcor_cols + cosine_cols + non_steady_cols + outlier_cols
    confounds = confounds[cols]

    return confounds, perc_outliers


def add_outlier_regressors(confounds, fd_thresh=0.5):
    """Adds outlier regressors based on framewise displacement to confounds."""

    fd = confounds['framewise_displacement']
    outlier_ixs = np.where(fd > fd_thresh)[0]
    outliers = np.zeros((len(fd), len(outlier_ixs)))
    outliers[outlier_ixs, np.arange(len(outlier_ixs))] = 1
    outlier_cols = [f'fd_outlier{i}' for i in range(len(outlier_ixs))]
    outliers = pd.DataFrame(outliers, columns=outlier_cols)
    confounds = pd.concat([confounds, outliers], axis=1)

    return confounds, outlier_cols


def load_df(layout, task, percs_outliers, df_query):
    """Load the DataFrame with the subject/session metadata for the mixed model."""

    df = layout.get_collections(task=task, level='session', types='scans',
                                merge=True).to_df()
    df = df.sort_values(['subject', 'session']).reset_index(drop=True)

    df['n_sessions'] = df.groupby('subject')['session'].transform('count')
    df['perc_outliers'] = percs_outliers
    df['acq_time'] = pd.to_datetime(df['acq_time'])
    df['time_diff'] = df['acq_time'] - df['acq_time'].min()
    df['time'] = df['time_diff'].dt.days / 30.437  # Convert to months
    df['time2'] = df['time'] ** 2

    df = df.query(df_query)

    return df[['subject', 'session', 'time', 'time2']]


def combine_save_mask_imgs(mask_imgs, output_dir, task, space, perc_thresh=0.5):
    """Combines brain masks across subjects and sessions and saves the result.

    Only voxels that are present in at least `perc_thresh` of all masks are
    included in the final mask."""

    mask_img = combine_mask_imgs(mask_imgs, perc_thresh)

    mask_file = save_img(mask_img, output_dir, task, space,
                         desc='brain', suffix='mask')

    return mask_img, mask_file


def combine_mask_imgs(mask_imgs, perc_thresh=0.5):
    """Combines brain masks across subjects and sessions.

    Only voxels that are present in at least `perc_thresh` of all masks are
    included in the final mask."""

    return binarize_img(mean_img(mask_imgs), threshold=perc_thresh)


def save_img(img, output_dir, task, space, desc, suffix):
    """Saves a NIfTI image to a file in the output directory."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.nii.gz'
    file = output_dir / filename
    img.to_filename(file)

    return file


def compute_beta_img(glm, conditions_plus, conditions_minus):
    """Computes a beta image from a fitted GLM for a given contrast."""

    design_matrices = glm.design_matrices_
    assert len(design_matrices) == 1
    design_matrix = design_matrices[0]

    contrast_values = np.zeros(design_matrix.shape[1])
    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)

    return glm.compute_contrast(contrast_values, output_type='effect_size')


def save_beta_img(beta_img, output_dir, subject, session, task, space,
                  contrast_label):
    """Saves a beta image to a NIfTI file in the output directory."""

    sub = f'sub-{subject}'
    ses = f'ses-{session}'
    tas = f'task-{task}'
    spc = f'space-{space}'
    des = f'desc-{contrast_label}'

    beta_dir = output_dir / sub / ses / 'func'
    beta_dir.mkdir(parents=True, exist_ok=True)
    beta_filename = f'{sub}_{ses}_{tas}_{spc}_{des}_effect_size.nii.gz'
    beta_file = beta_dir / beta_filename
    beta_img.to_filename(beta_file)


def fit_mixed_models(formula, dfs):
    """Fits mixed models for a list of DataFrames using the MixedModels package
    in Julia."""

    model_cmd = f"""
        using MixedModels
        using StatsBase
        using Suppressor

        function fit_mixed_model(df)
          fml = @formula({formula})
          mod = @suppress fit(MixedModel, fml, df)
          bs = mod.beta
          zs = mod.beta ./ mod.stderror
          residuals = StatsBase.residuals(mod)
        return bs, zs, residuals
        end
        
        function fit_mixed_models(dfs)
          map(fit_mixed_model, dfs)
        end"""
    fit_mixed_models_julia = jl.seval(model_cmd)

    return fit_mixed_models_julia(dfs)


def save_array_to_nifti(array, ref_img, voxel_ixs, output_dir, task, space,
                        desc, suffix):
    """Inserts an array into a NIfTI image and saves it to a file."""

    full_array = np.zeros(ref_img.shape)
    full_array[tuple(voxel_ixs.T)] = array
    img = nib.Nifti1Image(full_array, ref_img.affine)

    img_file = save_img(img, output_dir, task, space, desc, suffix)

    return img, img_file


def compute_acf(residuals_file, mask_file):
    """Computes paramters of the noise autocorrelation function using AFNI."""

    cmd = ['3dFWHMx',
           '-mask', mask_file,
           '-input', residuals_file,
           '-acf', 'NULL']
    res = run(cmd, stdout=PIPE, text=True, check=True)

    return res.stdout.split()[4:7]  # Select a, b, c parameters of ACF


def compute_cluster_threshold(acf, mask_file, sidedness, nn, voxel_thresh,
                              cluster_thresh, iter):

    cmd = ['3dClustSim',
           '-mask', mask_file,
           '-acf', *acf,
           '-pthr', str(voxel_thresh),
           '-athr', str(cluster_thresh),
           '-iter', str(iter)]
    res = run(cmd, stdout=PIPE, text=True, check=True)
    cluster_thresholds = parse_clustsim_output(res.stdout)

    return cluster_thresholds[sidedness][nn]


def parse_clustsim_output(output):
    """Parses the output of AFNI's 3dClustSim command into a dictionary."""

    lines = output.split('\n')
    ns_voxels = [float(line.split()[-1])
                 for line in lines
                 if line and not line.startswith('#')]

    results_dict = {}
    for sidedness in ['1-sided', '2-sided', 'bi-sided']:
        results_dict[sidedness] = {}
        for nn in [1, 2, 3]:
            results_dict[sidedness][f'NN{nn}'] = ns_voxels.pop(0)

    return results_dict


def save_clusters(img, voxel_threshold, cluster_threshold, output_dir, task,
                  space, contrast_label, suffix):
    """Saves clusters in a NIfTI image and a table of cluster statistics."""

    voxel_threshold_z = norm.ppf(1 - voxel_threshold / 2)  # p to z

    cluster_df, cluster_imgs = \
        get_clusters_table(img, voxel_threshold_z, cluster_threshold,
                           two_sided=True, return_label_maps=True)

    if cluster_imgs:

        save_df(cluster_df, output_dir, task, space, contrast_label,
                suffix=f'{suffix}-clusters')

        if len(cluster_imgs) == 1:

            save_img(cluster_imgs[0], output_dir, task, space, contrast_label,
                     suffix=f'{suffix}-clusters')

        else:

            save_img(cluster_imgs[0], output_dir, task, space, contrast_label,
                     suffix=f'{suffix}-clusters-pos')

            save_img(cluster_imgs[1], output_dir, task, space, contrast_label,
                     suffix=f'{suffix}-clusters-neg')


def save_df(df, output_dir, task, space, desc, suffix):
    """Saves a DataFrame with cluster statistics to a TSV file."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.tsv'
    file = output_dir / filename
    df.to_csv(file, sep='\t', index=False)

    return file


if __name__ == '__main__':
    main()
