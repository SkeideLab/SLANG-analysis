from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from bids import BIDSLayout, BIDSLayoutIndexer
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.surface import load_surf_data
from scipy.stats import ttest_1samp
from surfplot import Plot


def main():
    """The main function to run the univariate + multiavariate analyses."""

    task = 'language'
    space = 'fsaverage6'
    contrast_defs = {
        'audios-pseudo-noise': (['audios_pseudo'], ['audios_noise']),
        'images-pseudo-noise': (['images_pseudo'], ['images_noise']),
        'audios-words-pseudo': (['audios_words'], ['audios_pseudo']),
        'images-words-pseudo': (['images_words'], ['images_pseudo'])}
    fd_threshold = 1.0
    hrf_model = 'spm'
    session_pre = '01'
    session_post = '06'
    t_threshold = 2.0
    t_max = 8.0

    derivatives_dir = Path(__file__).parent.parent
    derivatives_dir = Path('/ptmp/aenge/slang/data/derivatives')
    bids_dir = derivatives_dir.parent
    fmriprep_dir = derivatives_dir / 'fmriprep'
    freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
    output_dir = derivatives_dir / 'nilearn_earli_group'

    pybids_dir = output_dir / 'pybids'
    pybids_dir.mkdir(parents=True, exist_ok=True)
    indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir], indexer=indexer,
                        database_path=pybids_dir)

    fsaverage = fetch_surf_fsaverage(mesh=space, data_dir=output_dir)
    inflated_files = fsaverage['infl_left'], fsaverage['infl_right']

    curv_files = fsaverage['curv_left'], fsaverage['curv_right']
    curv_maps = [load_surf_data(curv_file) for curv_file in curv_files]
    curv_map = np.concatenate(curv_maps)
    curv_map_bin = np.where(curv_map < 0.0, -1.0, 1.0)

    subjects_pre = sorted(layout.get_subjects(desc='preproc',
                                              session=session_pre))
    subjects_post = sorted(layout.get_subjects(desc='preproc',
                                               session=session_post))
    subjects = sorted(set(subjects_pre) & set(subjects_post))
    subjects = [s for s in subjects if s.startswith('SA')]
    pre_maps = {contrast_label: [] for contrast_label in contrast_defs}
    post_maps = {contrast_label: [] for contrast_label in contrast_defs}
    diff_maps = {contrast_label: [] for contrast_label in contrast_defs}
    for subject in subjects:
        print(f'\nPrecessing subject "{subject}"')
        pre_contrasts = compute_contrasts(layout, subject, session_pre, task,
                                          space, contrast_defs, fd_threshold,
                                          hrf_model)
        post_contrasts = compute_contrasts(layout, subject, session_post, task,
                                           space, contrast_defs, fd_threshold,
                                           hrf_model)
        for contrast_label in contrast_defs:
            pre_map = pre_contrasts[contrast_label].stat()
            post_map = post_contrasts[contrast_label].stat()
            diff_map = post_map - pre_map
            pre_maps[contrast_label].append(pre_map)
            post_maps[contrast_label].append(post_map)
            diff_maps[contrast_label].append(diff_map)

    maps = {session_pre: pre_maps,
            session_post: post_maps,
            'difference': diff_maps}

    for session_label, session_maps in maps.items():
        for contrast_label, contrast_maps in session_maps.items():
            contrast_maps = np.array(contrast_maps)
            t_map, p_map = ttest_1samp(contrast_maps, popmean=0.0)
            t_map_pos = np.where(t_map > t_threshold, t_map, 0.0)
            t_map_neg = np.where(t_map < -t_threshold, t_map, 0.0)
            plot = Plot(*inflated_files, views='lateral', size=(750, 300))
            _ = plot.add_layer(curv_map_bin, color_range=(-2.0, 5.0),
                               cmap='Greys', cbar=False)
            _ = plot.add_layer(t_map_pos, color_range=(t_threshold, t_max),
                               cmap='YlOrRd_r', cbar=False)
            _ = plot.add_layer(t_map_neg, color_range=(-t_max, -t_threshold),
                               cmap='YlGnBu', cbar=False)
            _ = plot.build()
            _ = plt.title(
                f'session {session_label}, contrast {contrast_label}')


def compute_contrasts(layout, subject, session, task, space, contrast_defs,
                      fd_threshold, hrf_model, roi_mask=None):
    """Computes auditory + visual univariate GLM contrasts for all sessions."""

    design_matrix, labels, estimates = \
        compute_glm(layout, subject, session, task, space, fd_threshold,
                    hrf_model, roi_mask)

    return {
        label: compute_psc_contrast(labels, estimates, design_matrix,
                                    conditions_plus, conditions_minus)
        for label, (conditions_plus, conditions_minus) in contrast_defs.items()}


def compute_glm(layout, subject, session, task, space, fd_threshold, hrf_model,
                roi_mask=None):
    """Fits a first-level GLM on the surface data for a single session."""

    event_cols = ['onset', 'duration', 'trial_type']
    events = layout.get_collections('run', 'events', subject=subject,
                                    session=session)[0]
    events = events.to_df()[event_cols]

    surf_files = layout.get('filename', scope='derivatives', subject=subject,
                            session=session, task=task, space=space,
                            suffix='bold', extension='.func.gii')
    assert len(surf_files) == 2, \
        'There must be exactly one preprocessed surface file per hemisphere'
    textures = [load_surf_data(f).T for f in surf_files]
    texture = np.concatenate(textures, axis=1)

    n_scans = texture.shape[0]
    start_time = layout.get_metadata(surf_files[0])['StartTime']
    t_r = layout.get_metadata(surf_files[0])['RepetitionTime']
    frame_times = start_time + t_r * np.arange(n_scans)

    confounds, sample_mask = get_confounds(layout, subject, session, task,
                                           fd_threshold)

    design_matrix = make_first_level_design_matrix(frame_times, events,
                                                   hrf_model, drift_model=None,
                                                   add_regs=confounds)

    texture = texture[sample_mask]
    design_matrix = design_matrix.iloc[sample_mask]
    condition_cols = set(design_matrix.columns) - set(confounds.columns)
    for col in condition_cols:
        col_max = design_matrix[col].max()
        if col_max <= 0.0:  # Remove trial regressor if scrubbed
            warn(f'Removing regressor \'{col}\' because it was scrubbed')
            design_matrix = design_matrix.drop(col, axis=1)
        else:
            scale_factor = 1.0 / col_max  # Scaling for PSC
            design_matrix[col] = design_matrix[col] * scale_factor

    mean_texture = texture.mean(axis=0)
    texture = 100.0 * (texture / mean_texture - 1.0)

    if roi_mask is not None:
        texture = texture[:, roi_mask]

    labels, estimates = run_glm(texture, design_matrix.values)

    return design_matrix, labels, estimates


def get_confounds(layout, subject, session, task, fd_threshold):
    """Loads confound regressors and sample mask (for scrubbing) from fMRIPrep
    outputs."""

    # It's necessary to get confound regressors via the *volumetric* fMRIPrep
    # outputs due to https://github.com/nilearn/nilearn/issues/3479 and
    # https://github.com/nilearn/nilearn/blob/b1fa2e/nilearn/interfaces/fmriprep/load_confounds_utils.py#L19
    vol_file = layout.get('filename', scope='derivatives', subject=subject,
                          session=session, task=task, space='T1w',
                          desc='preproc', extension='.nii.gz')[0]

    return load_confounds(
        vol_file, strategy=('motion', 'high_pass', 'wm_csf', 'scrub', 'compcor'),
        motion='basic', wm_csf='basic', scrub=0, fd_threshold=fd_threshold,
        std_dvars_threshold=None, compcor='anat_combined', n_compcor=6)


def compute_psc_contrast(labels, estimates, design_matrix,
                         conditions_plus=None, conditions_minus=None):
    """Computes a GLM-based contrast with betas in units of percent signal change."""

    contrast_values = np.zeros(design_matrix.shape[1])

    for col_ix, column in enumerate(design_matrix.columns):
        if conditions_plus is not None and column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if conditions_minus is not None and column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)

    return compute_contrast(labels, estimates, contrast_values)


if __name__ == '__main__':
    main()
