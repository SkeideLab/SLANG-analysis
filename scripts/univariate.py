from functools import partial
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
from nilearn.image import binarize_img, load_img, math_img, mean_img
from nilearn.reporting import get_clusters_table
from scipy.stats import norm

# Input parameters: File paths
BIDS_DIR = Path('/ptmp/aenge/SLANG')
DERIVATIVES_DIR = BIDS_DIR / 'derivatives'
FMRIPREP_DIR = DERIVATIVES_DIR / 'fmriprep'
PYBIDS_DIR = DERIVATIVES_DIR / 'pybids'
UNIVARIATE_DIR = DERIVATIVES_DIR / 'univariate'

# Input parameters: Inclusion/exclusiong criteria
FD_THRESHOLD = 0.5
DF_QUERY = 'subject.str.startswith("SA") & perc_outliers <= 0.25 & n_sessions >= 2'

# Inpupt parameters: First-level GLM
TASK = 'language'
SPACE = 'MNI152NLin2009cAsym'
BLOCKWISE = False
SMOOTHING_FWHM = 5.0
HRF_MODEL = 'glover + derivative + dispersion'
SAVE_RESIDUALS = True
CONTRASTS = {
    'audios-noise': (('audios_noise',), ()),
    'audios-pseudo': (('audios_pseudo',), ()),
    'audios-pseudo-minus-noise': (('audios_pseudo',), ('audios_noise',)),
    'audios-words': (('audios_words',), ()),
    'audios-words-minus-noise': (('audios_words',), ('audios_noise',)),
    'audios-words-minus-pseudo': (('audios_words',), ('audios_pseudo',)),
    'images-noise': (('images_noise',), ()),
    'images-pseudo': (('images_pseudo',), ()),
    'images-pseudo-minus-noise': (('images_pseudo',), ('images_noise',)),
    'images-words': (('images_words',), ()),
    'images-words-minus-noise': (('images_words',), ('images_noise',)),
    'images-words-minus-pseudo': (('images_words',), ('images_pseudo',))}
N_JOBS = 8

# Input parameters: Cluster correction
CLUSTSIM_SIDEDNESS = '2-sided'
CLUSTSIM_NN = 'NN1'  # Must be 'NN1' for Nilearn's `get_clusters_table`
CLUSTSIM_VOXEL_THRESHOLD = 0.001
CLUSTSIM_CLUSTER_THRESHOLD = 0.05
CLUSTSIM_ITER = 10000

# Input parameters: Group-level linear mixed models
FORMULA = 'beta ~ time + time2 + (time + time2 | subject)'


def main():
    """Main function for running the full session- and group-level analysis."""

    # Load BIDS structure
    layout = BIDSLayout(BIDS_DIR, derivatives=FMRIPREP_DIR,
                        database_path=PYBIDS_DIR)

    # Fit first-level GLM, separately for each subject and session
    glms, mask_imgs, percs_non_steady, percs_outliers, residuals_files = \
        run_glms(BIDS_DIR, FMRIPREP_DIR, PYBIDS_DIR, TASK, SPACE, BLOCKWISE,
                 FD_THRESHOLD, HRF_MODEL, SMOOTHING_FWHM, UNIVARIATE_DIR,
                 SAVE_RESIDUALS, N_JOBS)

    # Load metadata (subjects, sessions, time points) for mixed model
    meta_df, good_ixs = load_meta_df(layout, TASK, percs_non_steady,
                                     percs_outliers, DF_QUERY)
    meta_df_filename = f'task-{TASK}_space-{SPACE}_desc-metadata.tsv'
    meta_df_file = UNIVARIATE_DIR / meta_df_filename
    meta_df.to_csv(meta_df_file, sep='\t', index=False, float_format='%.5f')
    meta_df = meta_df.loc[good_ixs]

    # Combine brain masks
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]
    mask_img, mask_file = combine_save_mask_imgs(mask_imgs, UNIVARIATE_DIR,
                                                 TASK, SPACE)
    mask = mask_img.get_fdata().astype(np.int32)
    voxel_ixs = np.transpose(mask.nonzero())

    # Compute per-subject auto-correlation function (ACF) parameters and
    # average them for multiple comparison correction later on (see
    # https://discuss.afni.nimh.nih.gov/t/the-input-file-for-3dfwhmx/2528/2)
    residuals_files = [residuals_files[ix] for ix in good_ixs]
    acfs = [compute_acf(residuals_file, mask_file)
            for residuals_file in residuals_files]
    acf = np.mean(acfs, axis=0)
    print(f'ACF parameters (a, b, c): {acf}')

    # Compute FWE-corrected cluster threshold based on ACF parameters
    cluster_threshold = \
        compute_cluster_threshold(acf, mask_file, CLUSTSIM_SIDEDNESS,
                                  CLUSTSIM_NN, CLUSTSIM_VOXEL_THRESHOLD,
                                  CLUSTSIM_CLUSTER_THRESHOLD, CLUSTSIM_ITER)
    print(f'Cluster threshold: {cluster_threshold:.1f} voxels')

    # Loop over contrasts
    for contrast_label, (conditions_plus, conditions_minus) in CONTRASTS.items():

        # Compute beta images for each subject and session
        beta_imgs = [compute_beta_img(glm, conditions_plus, conditions_minus)
                     for ix, glm in enumerate(glms) if ix in good_ixs]
        betas = [load_img(img).get_fdata() for img in beta_imgs]
        betas = np.array(betas).squeeze()

        # Save beta images
        subjects = meta_df['subject']
        sessions = meta_df['session']
        for beta_img, subject, session in zip(beta_imgs, subjects, sessions):
            save_beta_img(beta_img, UNIVARIATE_DIR, subject, session, TASK,
                          SPACE, contrast_label)

        # Fit linear group-level linear mixed models using Julia
        print(f'Fitting mixed models for contrast "{contrast_label}"...')
        voxel_ixs = np.transpose(mask.nonzero())
        betas_per_voxel = [betas[:, x, y, z] for x, y, z in voxel_ixs]
        model_df = meta_df[['subject', 'session', 'perc_non_steady',
                            'perc_outliers', 'time', 'time2']]
        model_dfs = [model_df.assign(beta=betas) for betas in betas_per_voxel]
        res = fit_mixed_models(FORMULA, model_dfs)
        bs, zs = zip(*res)
        b0, b1, b2 = np.array(bs).T
        z0, z1, z2 = np.array(zs).T

        # Save unthresholded and thresholded statistical images to NIfTI files
        for array, suffix in zip([b0, b1, b2, z0, z1, z2],
                                 ['b0', 'b1', 'b2', 'z0', 'z1', 'z2']):

            img, file = save_array_to_nifti(array, mask_img, voxel_ixs,
                                            UNIVARIATE_DIR, TASK, SPACE,
                                            contrast_label, suffix)

            if suffix.startswith('z'):
                save_clusters(img, CLUSTSIM_VOXEL_THRESHOLD, cluster_threshold,
                              UNIVARIATE_DIR, TASK, SPACE, contrast_label, suffix)


def run_glms(bids_dir, fmriprep_dir, pybids_dir, task, space, blockwise,
             fd_threshold, hrf_model, smoothing_fwhm, output_dir,
             save_residuals, n_jobs):
    """Runs first-level GLM for all subjects and sessions."""

    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir,
                        database_path=pybids_dir)

    subjects_sessions = get_subjects_sessions(layout, task, space)

    run_glm_ = partial(run_glm, bids_dir, fmriprep_dir, pybids_dir, task,
                       space, blockwise, fd_threshold, hrf_model,
                       smoothing_fwhm, output_dir, save_residuals)

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


def run_glm(bids_dir, fmriprep_dir, pybids_dir, task, space, blockwise,
            fd_threshold, hrf_model, smoothing_fwhm, output_dir,
            save_residuals, subject, session):
    """Runs a first-level GLM for a given subject and session."""

    # bids_dir = BIDS_DIR
    # fmriprep_dir = FMRIPREP_DIR
    # pybids_dir = PYBIDS_DIR
    # task = TASK
    # space = SPACE
    # fd_threshold = FD_THRESHOLD
    # hrf_model = HRF_MODEL
    # smoothing_fwhm = SMOOTHING_FWHM
    # output_dir = UNIVARIATE_DIR
    # subject = 'SA01'
    # session = '01'

    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir,
                        database_path=pybids_dir)

    mask_img = load_mask_img(layout, subject, session, task, space)

    events = load_events(layout, subject, session, task)

    if blockwise:
        events['trial_type'] = [f'{trial_type}_{block_ix}'
                                for trial_type, block_ix
                                in zip(events['trial_type'], events.index)]

    func_img = load_func_img(layout, subject, session, task, space)

    frame_times = make_frame_times(layout, func_img)

    confounds, perc_non_steady, perc_outliers = \
        get_confounds(layout, subject, session, task, fd_threshold)

    design_matrix = make_first_level_design_matrix(frame_times,
                                                   events=events,
                                                   hrf_model=hrf_model,
                                                   drift_model=None,
                                                   high_pass=None,
                                                   add_regs=confounds)

    glm_small = FirstLevelModel(smoothing_fwhm=smoothing_fwhm,
                                mask_img=mask_img, minimize_memory=False)
    glm_small.fit(run_imgs=func_img, design_matrices=[design_matrix])

    if not save_residuals:

        return glm_small, mask_img, perc_outliers, perc_non_steady

    else:

        glm_big = FirstLevelModel(smoothing_fwhm=smoothing_fwhm,
                                  mask_img=mask_img,
                                  minimize_memory=False)
        glm_big.fit(run_imgs=func_img, design_matrices=[design_matrix])

        func_dir = output_dir / f'sub-{subject}' / f'ses-{session}' / 'func'
        func_dir.mkdir(parents=True, exist_ok=True)

        residuals_filename = f'sub-{subject}_ses-{session}_task-{task}_space-{space}_desc-residuals.nii.gz'
        residuals_file = func_dir / residuals_filename
        residuals = glm_big.residuals[0]
        residuals.to_filename(residuals_file)

        return glm_small, mask_img, perc_outliers, perc_non_steady, residuals_file


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


def get_confounds(layout, subject, session, task, fd_threshold=0.5):
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
    n_non_steady = len(non_steady_cols)
    n_volumes = len(confounds)
    perc_non_steady = n_non_steady / n_volumes
    print(f'Found {n_non_steady} non-steady-state volumes ' +
          f'({perc_non_steady * 100:.1f}%) for subject {subject}, session {session}')

    confounds, outlier_cols = add_outlier_regressors(confounds, fd_threshold)

    n_outliers = len(outlier_cols)
    perc_outliers = n_outliers / n_volumes
    print(f'Found {n_outliers} outlier volumes ({perc_outliers * 100:.1f}%) ' +
          f'for subject {subject}, session {session}')

    cols = hmp_cols + compcor_cols + cosine_cols + non_steady_cols + outlier_cols
    confounds = confounds[cols]

    return confounds, perc_non_steady, perc_outliers


def add_outlier_regressors(confounds, fd_threshold=0.5):
    """Adds outlier regressors based on framewise displacement to confounds."""

    fd = confounds['framewise_displacement']
    outlier_ixs = np.where(fd > fd_threshold)[0]
    outliers = np.zeros((len(fd), len(outlier_ixs)))
    outliers[outlier_ixs, np.arange(len(outlier_ixs))] = 1
    outlier_cols = [f'fd_outlier{i}' for i in range(len(outlier_ixs))]
    outliers = pd.DataFrame(outliers, columns=outlier_cols)
    confounds = pd.concat([confounds, outliers], axis=1)

    return confounds, outlier_cols


def load_meta_df(layout, task, percs_outliers, percs_non_steady, df_query):
    """Load the DataFrame with the subject/session metadata for the mixed model."""

    df = layout.get_collections(task=task, level='session', types='scans',
                                merge=True).to_df()
    df = df.sort_values(['subject', 'session']).reset_index(drop=True)

    df['n_sessions'] = df.groupby('subject')['session'].transform('count')
    df['perc_non_steady'] = percs_non_steady
    df['perc_outliers'] = percs_outliers
    df['acq_time'] = pd.to_datetime(df['acq_time'])
    df['time_diff'] = df['acq_time'] - df['acq_time'].min()
    df['time'] = df['time_diff'].dt.days / 30.437  # Convert to months
    df['time2'] = df['time'] ** 2

    good_df = df.query(df_query)
    good_ixs = good_df.index
    df['include'] = False
    df.loc[good_ixs, 'include'] = True

    return df, good_ixs


def combine_save_mask_imgs(mask_imgs, output_dir, task, space,
                           perc_threshold=0.5):
    """Combines brain masks across subjects and sessions and saves the result.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    mask_img = combine_mask_imgs(mask_imgs, perc_threshold)

    mask_file = save_img(mask_img, output_dir, task, space,
                         desc='brain', suffix='mask')

    return mask_img, mask_file


def combine_mask_imgs(mask_imgs, perc_threshold=0.5):
    """Combines brain masks across subjects and sessions.

    Only voxels that are present in at least `perc_threshold` of all masks are
    included in the final mask."""

    return binarize_img(mean_img(mask_imgs), threshold=perc_threshold)


def save_img(img, output_dir, task, space, desc, suffix,
             subject=None, session=None):
    """Saves a NIfTI image to a file in the output directory."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.nii.gz'

    if session:
        filename = f'ses-{session}_{filename}'

    if subject:
        filename = f'sub-{subject}_{filename}'

    file = output_dir / filename
    img.to_filename(file)

    return file


def compute_acf(residuals_file, mask_file):
    """Computes parameters of the noise autocorrelation function using AFNI."""

    cmd = ['3dFWHMx',
           '-mask', mask_file,
           '-input', residuals_file,
           '-acf', 'NULL']
    res = run(cmd, stdout=PIPE, text=True, check=True)

    acf = res.stdout.split()[4:7]  # Select a, b, c parameters of ACF

    return [float(param) for param in acf]


def compute_cluster_threshold(acf, mask_file, sidedness, nn, voxel_threshold,
                              cluster_threshold, iter):
    """Computes the FWE-corrected cluster size threshold using AFNI."""

    cmd = ['3dClustSim',
           '-mask', mask_file,
           '-acf', *[str(param) for param in acf],
           '-pthr', str(voxel_threshold),
           '-athr', str(cluster_threshold),
           '-iter', str(iter)]
    res = run(cmd, stdout=PIPE, text=True, check=True)
    cluster_thresholds = parse_clustsim_output(res.stdout)

    return cluster_thresholds[sidedness][nn]


def parse_clustsim_output(output):
    """Parses the output of AFNI's `3dClustSim` command into a dictionary."""

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
    """Fits mixed models for a list of DataFrames using the `MixedModels`
    package in Julia."""

    model_cmd = f"""
        using MixedModels
        using Suppressor

        function fit_mixed_model(df)
          fml = @formula({formula})
          mod = @suppress fit(MixedModel, fml, df)
          bs = mod.beta
          zs = mod.beta ./ mod.stderror
        return bs, zs
        end
        
        function fit_mixed_models(dfs)
          map(fit_mixed_model, dfs)
        end"""
    fit_mixed_models_julia = jl.seval(model_cmd)

    return fit_mixed_models_julia(dfs)


def save_array_to_nifti(array, ref_img, voxel_ixs, output_dir, task, space,
                        desc, suffix, subject=None, session=None):
    """Inserts a NumPy array into a NIfTI image and saves it to a file."""

    full_array = np.zeros(ref_img.shape)
    full_array[tuple(voxel_ixs.T)] = array
    img = nib.Nifti1Image(full_array, ref_img.affine)

    img_file = save_img(img, output_dir, task, space, desc, suffix)

    return img, img_file


def save_clusters(img, voxel_threshold, cluster_threshold, output_dir, task,
                  space, contrast_label, suffix):
    """Finds clusters in a z-map and saves them as a table and NIfTI image."""

    voxel_threshold_z = norm.ppf(1 - voxel_threshold / 2)  # p to z

    cluster_df, cluster_imgs = \
        get_clusters_table(img, voxel_threshold_z, cluster_threshold,
                           two_sided=True, return_label_maps=True)

    has_pos_clusters = any(cluster_df['Peak Stat'] > 0)
    has_neg_clusters = any(cluster_df['Peak Stat'] < 0)

    if has_pos_clusters:

        if has_neg_clusters:

            neg_ixs = cluster_df['Peak Stat'] < 0
            cluster_df.loc[neg_ixs, 'Cluster ID'] = \
                '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

            cluster_img = math_img('img_pos - img_neg',
                                   img_pos=cluster_imgs[0],
                                   img_neg=cluster_imgs[1])

        else:

            cluster_img = cluster_imgs[0]

    elif has_neg_clusters:

        neg_ixs = cluster_df['Peak Stat'] < 0
        cluster_df.loc[neg_ixs, 'Cluster ID'] = \
            '-' + cluster_df.loc[neg_ixs, 'Cluster ID'].astype(str)

        cluster_img = math_img('-img', img=cluster_imgs[0])

    else:

        cluster_img = math_img('img - img', img=img)

    save_df(cluster_df, output_dir, task, space, contrast_label,
            suffix=f'{suffix}-clusters')

    save_img(cluster_img, output_dir, task, space, contrast_label,
             suffix=f'{suffix}-clusters')


def save_df(df, output_dir, task, space, desc, suffix):
    """Saves a DataFrame to a TSV file."""

    filename = f'task-{task}_space-{space}_desc-{desc}_{suffix}.tsv'
    file = output_dir / filename
    df.to_csv(file, sep='\t', index=False, float_format='%.5f')

    return file


if __name__ == '__main__':
    main()
