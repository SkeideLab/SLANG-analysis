from os import chdir, getcwd
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer
from datalad_container.containers_run import ContainersRun
from mne.datasets import fetch_fsaverage
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.surface import load_surf_data
from surfplot import Plot


def main():
    """The main function to run the univariate + multiavariate analyses."""

    task = 'language'
    space = 'fsnative'
    contrast_defs = {
        'audios-pseudo-noise': (['audios_pseudo'], ['audios_noise']),
        'images-pseudo-noise': (['images_pseudo'], ['images_noise']),
        'audios-words-pseudo': (['audios_words'], ['audios_pseudo']),
        'images-words-pseudo': (['images_words'], ['images_pseudo'])}
    fd_threshold = 2.4
    hrf_model = 'spm'
    glasser_rois = {'dpSTS': [129],
                    'VWFA_a': [136, 138]}
    roi_plot_colors = ['#6A3D9A', '#FF7F00']
    roi_plot_subject = 'SA15'

    derivatives_dir = Path(__file__).parent.parent
    derivatives_dir = Path('/ptmp/aenge/slang/data/derivatives')
    bids_dir = derivatives_dir.parent
    fmriprep_dir = derivatives_dir / 'fmriprep'
    freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
    output_dir = derivatives_dir / 'nilearn'

    pybids_dir = derivatives_dir / 'pybids'
    pybids_dir.mkdir(exist_ok=True)
    indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir], indexer=indexer,
                        database_path=pybids_dir)

    _ = fetch_fsaverage(freesurfer_dir)

    dfs = []
    subjects = sorted(layout.get_subjects(desc='preproc'))
    for subject in subjects:
        print(f'\nPrecessing subject "{subject}"')

        sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
        for session in sessions:
            print(f'Precessing session "{session}"')

            for hemi in ['L', 'R']:
                print(f'Precessing hemisphere "{hemi}"')

                contrasts = compute_contrasts(layout, subject, session, task,
                                              space, hemi, contrast_defs,
                                              fd_threshold, hrf_model)

                for roi_label, roi_ixs in glasser_rois.items():
                    print(f'Precessing ROI "{roi_label}"')

                    roi_mask = make_roi_mask(derivatives_dir, freesurfer_dir,
                                             subject, hemi, roi_ixs)
                    n_vertices = roi_mask.sum()

                    for label, contrast in contrasts.items():
                        print(f'Precessing contrast "{label}"')

                        effects = contrast.effect_size()[roi_mask]
                        variances = contrast.effect_variance()[roi_mask]

                        mean_effect = effects.mean()
                        mean_variance = variances.mean()
                        mean_sd = np.sqrt(mean_variance)
                        voxels_sd = np.std(effects)

                        df = pd.DataFrame({'subject': f'sub-{subject}',
                                           'session': f'ses-{session}',
                                           'hemi': hemi,
                                           'roi': roi_label,
                                           'n_vertices': n_vertices,
                                           'contrast': label,
                                           'effect': mean_effect,
                                           'mean_sd': mean_sd,
                                           'voxels_sd': voxels_sd},
                                          index=[0])
                        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df_dir = output_dir / 'sub-group'
    save_df(df, df_dir, subject='group', task=task,
            space=space, desc='univariate')

    print(f'Creating ROI plot for example subject "{roi_plot_subject}"')
    roi_masks = []
    for roi_ixs in glasser_rois.values():
        roi_masks_hemis = []
        for hemi in ['L', 'R']:
            roi_mask = make_roi_mask(derivatives_dir, freesurfer_dir,
                                     roi_plot_subject, hemi, roi_ixs)
            roi_masks_hemis.append(roi_mask)
        roi_mask = np.concatenate(roi_masks_hemis)
        roi_masks.append(roi_mask)
    plot = make_roi_plot(layout, roi_plot_subject, roi_masks, roi_plot_colors)
    plot_dir = output_dir / f'sub-{roi_plot_subject}'
    plot_dir.mkdir(exist_ok=True)
    plot_file = plot_dir / \
        f'sub-{roi_plot_subject}_task-{task}_space-{space}_desc-rois_plot.png'
    plot.savefig(plot_file, dpi=500, bbox_inches='tight')


def compute_contrasts(layout, subject, session, task, space, hemi,
                      contrast_defs, fd_threshold, hrf_model, roi_mask=None):
    """Computes auditory + visual univariate GLM contrasts for all sessions."""

    design_matrix, labels, estimates = \
        compute_glm(layout, subject, session, task, hemi, space, fd_threshold,
                    hrf_model, roi_mask, use_single_trials=False)

    return {
        label: compute_psc_contrast(labels, estimates, design_matrix,
                                    conditions_plus, conditions_minus)
        for label, (conditions_plus, conditions_minus) in contrast_defs.items()}


def compute_glm(layout, subject, session, task, hemi, space, fd_threshold,
                hrf_model, roi_mask=None, use_single_trials=False):
    """Fits a first-level GLM on the surface data for a single session."""

    event_cols = ['onset', 'duration', 'trial_type']
    events = layout.get_collections('run', 'events', subject=subject,
                                    session=session)[0]
    events = events.to_df()[event_cols]

    if use_single_trials:
        events = events_to_single_trials(events)

    surf_files = layout.get('filename', scope='derivatives', subject=subject,
                            session=session, task=task, hemi=hemi, space=space,
                            suffix='bold', extension='.func.gii')
    assert len(surf_files) == 1, \
        'There must be exactly one preprocessed surface file'
    texture = load_surf_data(surf_files[0]).T

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


def events_to_single_trials(events):
    """Converts the BIDS events dataframe to one separate condition per trial.
    See https://nilearn.github.io/dev/auto_examples/07_advanced/plot_beta_series.html"""

    new_events = events.copy()
    conditions = new_events['trial_type'].unique()
    condition_counter = {c: 0 for c in conditions}
    for trial_ix, trial in new_events.iterrows():
        condition = trial['trial_type']
        condition_counter[condition] += 1
        trial_name = f'{condition}_{condition_counter[condition]:02d}'
        new_events.loc[trial_ix, 'trial_type'] = trial_name

    return new_events


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


def make_roi_mask(derivatives_dir, freesurfer_dir, subject, hemi, roi_ixs):
    """Creates a ROI map for one or more Glasser ROIs in fsnative space."""

    freesurfer_hemi = 'lh' if hemi == 'L' else 'rh'
    atlas_file = surf_to_surf(derivatives_dir, freesurfer_dir,
                              freesurfer_hemi, source_subject='fsaverage',
                              target_subject=f'sub-{subject}',
                              sval_filename='HCPMMP1.annot')

    atlas_map = load_surf_data(atlas_file)

    if isinstance(roi_ixs, (int, float, str)):
        roi_ixs = [int(roi_ixs)]
    else:
        roi_ixs = [int(ix) for ix in roi_ixs]

    return np.sum([atlas_map == ix for ix in roi_ixs], axis=0, dtype=bool)


def surf_to_surf(derivatives_dir, freesurfer_dir, freesurfer_hemi,
                 source_subject, target_subject, sval_filename,
                 tval_filename=None):
    """Uses the FreeSurfer container to convert surface data between different
    subjects; e.g., to convert a parcellation (`.annot`) defined in fsaverage
    space to the fsnative space of a single subject."""

    freesurfer_rel_dir = Path(freesurfer_dir).relative_to(derivatives_dir)

    if Path(sval_filename).suffix == '.annot':
        sval_flag = ['--sval-annot', sval_filename]
    else:
        sval_flag = ['--sval', sval_filename]

    if tval_filename is None:
        tval_filename = sval_filename

    output_file = freesurfer_dir / target_subject / \
        f'label/{freesurfer_hemi}.{sval_filename}'
    if not output_file.exists():

        output_rel_file = freesurfer_rel_dir / target_subject / \
            f'label/{freesurfer_hemi}.{tval_filename}'

        cmd = ['mri_surf2surf',
               '--srcsubject', source_subject,
               '--trgsubject', target_subject,
               '--hemi', freesurfer_hemi,
               *sval_flag,
               '--tval', sval_filename,
               '--sd', freesurfer_rel_dir]
        cmd = ' '.join([str(elem) for elem in cmd])
        cmd = f'-c \'{cmd}\''  # See https://stackoverflow.com/a/62313159

        # # To use Docker instead of Singularity:
        # from os import environ
        # environ['REPRONIM_USE_DOCKER'] = 'TRUE'

        current_dir = getcwd()
        chdir(derivatives_dir)

        cr = ContainersRun()
        container_name = 'code/containers/neurodesk-freesurfer'
        inputs = [str(freesurfer_rel_dir / source_subject),
                  str(freesurfer_rel_dir / target_subject)]
        outputs = [str(output_rel_file)]
        cr(cmd, container_name, inputs=inputs, outputs=outputs,
           message='Convert surface annotations from ' +
                   f'{source_subject} to {target_subject}',
           explicit=True)

        chdir(current_dir)

    else:
        print(f'Output surface file "{output_file}" exists, nothing to do')

    return output_file


def make_roi_plot(layout, subject, roi_masks, roi_colors):
    """Plots a statistical and/or ROI map(s) on the inflated FreeSurfer surface."""

    inflated_files = sorted(layout.get('filename', subject=subject,
                                       suffix='inflated',
                                       extension='surf.gii'))
    plot = Plot(*inflated_files, views='lateral', size=(1000, 300), zoom=1.8)

    curv_files = sorted(layout.get('filename', subject=subject,
                                   extension='curv'))
    curv_map = np.concatenate([load_surf_data(f) for f in curv_files])
    curv_map_sign = np.sign(curv_map)
    plot.add_layer(curv_map_sign, cmap='Greys', color_range=[-1.0, 4.0],
                   cbar=False)

    color_range = (1.0, len(roi_masks))
    cmap = colors.ListedColormap(roi_colors)
    for roi_ix, roi_mask in enumerate(roi_masks):
        roi_multiplier = roi_ix + 1
        roi_map = roi_mask * roi_multiplier
        plot.add_layer(roi_map, cmap=cmap, alpha=0.8,
                       color_range=color_range, cbar=False)

    return plot.build()


def save_df(df, output_dir, subject, task, space, desc):
    """Saves a pandas DataFrame to a CSV file."""

    filename = f'sub-{subject}_task-{task}_space-{space}_desc-{desc}_df.csv'
    file = output_dir / filename
    df.to_csv(file, index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
