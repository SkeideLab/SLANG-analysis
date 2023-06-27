"""Influence of learning to read on multi-modal language regions (pSTS).

Using univariate (GLM) and multivariate (pattern stability) analysis on the
cortical surface of fMRIPrep-preprocessed data.

Analysis steps / goals:

1. Identify vertices in the posterior superior temporal sulcus ROI (pSTS;
   defined in the Glasser et al., 2016 atlas) that are sensitve to the the
   contrast of *auditory* pseudowords minus noise. I refer to this as the
   "fROI" (functional region of interest). It's computed by taking the top *n*
   (here 500) vertices from a fixed-effects model accross all sessions.

2. Compute the percent signal change (PSC) for the contrast of *visual*
   pseudowords minus noise in the fROI, separately for each session. We execpt
   the PSC to increase as children learn to connect written syllables to their
   speech sounds, i.e., that learning to read makes them recruit pSTS regions
   that had already been tuned to spoken language.

3. For each condition, compute the multivariate pattern stability (PS) in the
   fROI, separately for each session. We expect the PS to increase as children
   learn to read, especially for visual pseudowords. PS is defined as the
   average correlation between the fROI pattern for each pair of trials from
   the same condition.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer
from datalad.api import Dataset
from mne.datasets import fetch_fsaverage
from nilearn.glm.contrasts import (_compute_fixed_effects_params,
                                   compute_contrast)
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.plotting.cm import cold_hot
from nilearn.surface import load_surf_data
from surfplot import Plot


def main():
    """The main function to run the univariate + multiavariate analyses."""

    task = 'language'
    space = 'fsnative'
    fd_threshold = 2.4
    hrf_model = 'spm'
    roi_ixs = 129
    perc_top_vertices = 0.25

    derivatives_dir = Path(__file__).parent.parent
    bids_dir = derivatives_dir.parent
    fmriprep_dir = derivatives_dir / 'fmriprep'
    freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
    output_dir = derivatives_dir / 'nilearn'

    indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
    layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir], indexer=indexer)

    fsaverage_dir = fetch_fsaverage(freesurfer_dir)

    psc_dfs = []
    distance_dfs = []
    stability_dfs = []
    for subject in ['SA15']:  # layout.get_subjects(desc='preproc'):

        print(f'\nPrecessing subject sub-{subject}\n')

        t_maps = []
        roi_maps = []
        froi_maps = []
        for hemi in ['L', 'R']:

            aud_contrasts, vis_contrasts = \
                compute_session_contrasts(layout, subject, task, hemi, space,
                                          fd_threshold, hrf_model)

            psc_map, var_map, t_map = \
                compute_fixed_effects(aud_contrasts.values())
            t_maps.append(t_map)

            roi_map = make_glasser_roi_map(derivatives_dir, freesurfer_dir,
                                           subject, hemi, roi_ixs)
            roi_maps.append(roi_map)

            froi_map = make_froi_map(t_map, roi_map, perc_top_vertices)
            froi_maps.append(froi_map)
            n_froi_vertices = froi_map.sum()

            for session, contrast in vis_contrasts.items():
                psc = contrast.effect[0][froi_map].mean()
                psc_df = pd.DataFrame({'subject': f'sub-{subject}',
                                       'session': f'ses-{session}',
                                       'hemi': hemi,
                                       'n_froi_vertices': n_froi_vertices,
                                       'psc': psc},
                                      index=[0])
                psc_dfs.append(psc_df)

            distance_df = compute_distances(subject, hemi, aud_contrasts,
                                            vis_contrasts, roi_map=None)
            distance_dfs.append(distance_df)

            stability_df = compute_pattern_stability(layout, subject, task,
                                                     hemi, space, fd_threshold,
                                                     hrf_model, roi_map)
            stability_dfs.append(stability_df)

        t_map = np.concatenate(t_maps, axis=0)
        roi_map = np.concatenate(roi_maps, axis=0)
        froi_map = np.concatenate(froi_maps, axis=0)
        _ = make_surfplot(layout, subject, t_map, roi_map, froi_map,
                          cbar_label='Spoken pseudowords\nminus noise ($t$)',
                          vmin=-11.0, vmax=11.0)

        output_sub_dir = output_dir / f'nilearn/sub-{subject}'
        output_sub_dir.mkdir(exist_ok=True, parents=True)
        plot_filename = f'sub-{subject}_task-{task}_space-{space}_desc-aud-pseudo-minus-noise-psts_plot.png'
        plot_file = output_sub_dir / plot_filename
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')

    output_group_dir = output_dir / 'nilearn/sub-group'
    output_group_dir.mkdir(exist_ok=True, parents=True)

    psc_df = pd.concat(psc_dfs)
    save_df(psc_df, output_group_dir, subject='group', task=task, space=space,
            desc='vis-pseudo-minus-noise-psts')

    distance_df = pd.concat(distance_dfs)
    save_df(distance_df, output_group_dir, subject='group', task=task,
            space=space, desc='vis-aud-distance-psts')

    stability_df = pd.concat(stability_dfs)
    save_df(stability_df, output_group_dir, subject='group', task=task,
            space=space, desc='pattern-stability-psts')


def compute_session_contrasts(layout, subject, task, hemi, space, fd_threshold,
                              hrf_model):
    """Computes auditory + visual univariate GLM contrasts for all sessions."""

    aud_contrasts = {}
    vis_contrasts = {}

    sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
    for session in sessions:

        design_matrix, labels, estimates = compute_glm(layout, subject,
                                                       session, task, hemi,
                                                       space, fd_threshold,
                                                       hrf_model,
                                                       use_single_trials=False)

        aud_contrasts[session] = \
            compute_psc_contrast(labels, estimates, design_matrix,
                                 conditions_plus=['audios_pseudo'],
                                 conditions_minus=['audios_noise'])

        vis_contrasts[session] = \
            compute_psc_contrast(labels, estimates, design_matrix,
                                 conditions_plus=['images_pseudo'],
                                 conditions_minus=['images_noise'])

    return aud_contrasts, vis_contrasts


def compute_glm(layout, subject, session, task, hemi, space, fd_threshold,
                hrf_model, use_single_trials=False):
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
        if design_matrix[col].sum() == 0.0:  # Remove trial regressor if scrubbed
            design_matrix = design_matrix.drop(col, axis=1)
        else:
            scale_factor = 1.0 / max(design_matrix[col])  # Scaling for PSC
            design_matrix[col] = design_matrix[col] * scale_factor

    mean_texture = texture.mean(axis=0)
    texture = 100.0 * (texture / mean_texture - 1.0)

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
                         conditions_plus, conditions_minus):
    """Computes a GLM-based contrast with betas in units of percent signal change."""

    contrast_values = np.zeros(design_matrix.shape[1])

    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)

    return compute_contrast(labels, estimates, contrast_values)


def compute_fixed_effects(contrasts):
    """Computes a fixed-effects model from a list of first level contrasts."""

    effects = [contrast.effect for contrast in contrasts]
    variances = [contrast.variance for contrast in contrasts]

    psc_maps, var_maps, t_maps = _compute_fixed_effects_params(
        effects, variances, precision_weighted=False)

    return psc_maps[0], var_maps[0], t_maps[0]


def make_glasser_roi_map(derivatives_dir, freesurfer_dir, subject, hemi,
                         roi_ixs):
    """Creates a ROI map for one or more Glasser ROIs in fsnative space."""

    derivatives_dataset = Dataset(derivatives_dir)
    freesurfer_hemi = 'lh' if hemi == 'L' else 'rh'
    atlas_file = surf_to_surf(derivatives_dataset, freesurfer_dir,
                              freesurfer_hemi, source_subject='fsaverage',
                              target_subject=f'sub-{subject}',
                              sval_filename='HCPMMP1.annot')

    atlas_map = load_surf_data(atlas_file)

    if isinstance(roi_ixs, (int, float, str)):
        roi_ixs = [int(roi_ixs)]
    else:
        roi_ixs = [int(ix) for ix in roi_ixs]

    return np.sum([atlas_map == roi_ix for roi_ix in roi_ixs], axis=0)


def surf_to_surf(derivatives_dataset, freesurfer_dir, freesurfer_hemi,
                 source_subject, target_subject, sval_filename,
                 tval_filename=None):
    """Uses the FreeSurfer container to convert surface data between different
    subjects; e.g., to convert a parcellation (`.annot`) defined in fsaverage
    space to the fsnative space of a single subject."""

    freesurfer_dir = Path(freesurfer_dir)

    if Path(sval_filename).suffix == '.annot':
        sval_flag = ['--sval-annot', sval_filename]
    else:
        sval_flag = ['--sval', sval_filename]

    if tval_filename is None:
        tval_filename = sval_filename

    input_file = freesurfer_dir / \
        f'{source_subject}/label/{freesurfer_hemi}.{sval_filename}'
    output_file = freesurfer_dir / \
        f'{target_subject}/label/{freesurfer_hemi}.{sval_filename}'

    if not output_file.exists():

        cmd = ['mri_surf2surf',
               '--srcsubject', source_subject,
               '--trgsubject', target_subject,
               '--hemi', freesurfer_hemi,
               *sval_flag,
               '--tval', sval_filename,
               '--sd', freesurfer_dir]
        cmd = ' '.join([str(elem) for elem in cmd])
        cmd = f'-c \'{cmd}\''  # See https://stackoverflow.com/a/62313159

        # # To use Docker instead of Singularity:
        # from os import environ
        # environ['REPRONIM_USE_DOCKER'] = 'TRUE'

        derivatives_dataset.containers_run(
            cmd, container_name='code/containers/neurodesk-freesurfer',
            inputs=[str(input_file)], outputs=[str(output_file)],
            message=f'Convert surface annotations from {source_subject} to {target_subject}',
            explicit=True)

    else:
        print(f'Output surface file "{output_file}" exists, nothing to do')

    return output_file


def make_froi_map(t_map, roi_map, perc_top_vertices):
    """Create functional region of interest based on t-values in the ROI."""

    # Get number of top vertices to extract from the ROI
    n_top_vertices = int(np.round(perc_top_vertices * np.sum(roi_map)))

    # Get indices of top vertices (i.e., vertices with highest t-values)
    t_map_roi = t_map * roi_map
    top_ixs = np.argsort(t_map_roi)[::-1][:n_top_vertices]

    # Create fROI map (i.e., 1s at the top vertices and 0s elsewhere)
    froi_map = np.zeros_like(t_map_roi, dtype='int')
    froi_map[top_ixs] = 1

    return froi_map


def make_surfplot(layout, subject, stat_map=None, roi_map_1=None,
                  roi_map_2=None, add_curv=True, views='lateral',
                  size=(1000, 300), zoom=2.0, cmap=cold_hot, cbar_label=None,
                  vmin=-2.0, vmax=2.0):
    """Plots a statistical and/or ROI map(s) on the inflated FreeSurfer surface."""

    inflated_files = sorted(layout.get('filename', subject=subject,
                                       suffix='inflated',
                                       extension='surf.gii'))
    plot = Plot(*inflated_files, views=views, size=size, zoom=zoom)

    if add_curv:
        curv_files = sorted(layout.get('filename', subject=subject,
                                       extension='curv'))
        curv_map = np.concatenate([load_surf_data(f) for f in curv_files])
        curv_map_sign = np.sign(curv_map)
        _ = plot.add_layer(curv_map_sign, cmap='Greys',
                           color_range=[-8.0, 4.0], cbar=False)

    # TODO: Use the `cmcrameri` or `palettable` package for the manauga
    # colormap once they have been updated to Scientific Color Maps 8.
    # Remember to also remove Nilearn's `cold_hot`.
    from matplotlib.colors import LinearSegmentedColormap
    cmap_file = '/Users/alexander/Downloads/ScientificColourMaps8/managua/managua.txt'
    cmap_data = np.loadtxt(cmap_file)
    cmap = LinearSegmentedColormap.from_list(
        'managua_r', np.flip(cmap_data, axis=0))

    if stat_map is not None:
        _ = plot.add_layer(stat_map, cmap=cmap, color_range=[vmin, vmax],
                           cbar=True, cbar_label=cbar_label)

    if roi_map_1 is not None:
        _ = plot.add_layer(roi_map_1, cmap='Greys_r',
                           as_outline=True, cbar=False)

    if roi_map_2 is not None:
        _ = plot.add_layer(roi_map_2, cmap='brg', as_outline=True, cbar=False)

    return plot.build()


def compute_distances(subject, hemi, aud_contrasts, vis_contrasts,
                      roi_map=None):
    """Compute pairwise cosine/correlation distance between condition beta maps."""

    sessions = aud_contrasts.keys()

    corrs = []
    for aud_contrast, vis_contrast in zip(aud_contrasts.values(),
                                          vis_contrasts.values()):

        if roi_map is None:
            aud_map = aud_contrast.effect
            vis_map = vis_contrast.effect
        else:
            aud_map = aud_contrast.effect[roi_map]
            vis_map = vis_contrast.effect[roi_map]

        corr = np.corrcoef(aud_map, vis_map)[0, 1]
        corrs.append(corr)

    return pd.DataFrame({'subject': f'sub-{subject}',
                         'session': [f'ses-{ses}' for ses in sessions],
                         'hemi': hemi,
                         'similarity': corrs})


def compute_pattern_stability(layout, subject, task, hemi, space, fd_threshold,
                              hrf_model, roi_map=None):
    """Compute single-trial pattern stability for each session and condition.
    Pattern stability is defined as the average correlation between all pairs
    of trials within a condition."""

    conditions = ['audios_pseudo', 'audios_noise',
                  'images_pseudo', 'images_noise']
    sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
    corrs = {condition: {} for condition in conditions}
    for session in sessions:

        design_matrix, labels, estimates = compute_glm(layout, subject,
                                                       session, task, hemi,
                                                       space, fd_threshold,
                                                       hrf_model,
                                                       use_single_trials=True)

        design_cols = design_matrix.columns.str
        for condition in conditions:

            trial_ixs = np.where(design_cols.startswith(condition))[0]
            trial_pairs = list(combinations(trial_ixs, 2))

            n_pairs = len(trial_pairs)
            print(f'Correlating {n_pairs} pairs of trials for "{condition}"')

            condition_corrs = []
            for ixs in trial_pairs:
                corr = correlate_trials(design_matrix, ixs,
                                        labels, estimates, roi_map)
                condition_corrs.append(corr)

            corrs[condition][session] = np.mean(condition_corrs)

    df = pd.DataFrame(corrs)
    df = df.reset_index(names='session')
    df['session'] = 'ses-' + df['session'].astype(str)
    df.insert(0, 'subject', f'sub-{subject}')
    df.insert(2, 'hemi', hemi)

    return pd.melt(df, id_vars=['subject', 'session', 'hemi'],
                   var_name='condition', value_name='stability')


def correlate_trials(design_matrix, ixs, labels, estimates, roi_map=None):
    """Computes the whole-brain or ROI pattern correlation between a pair of
    trials in the design matrix."""

    effect_maps = []
    for ix in ixs:

        con_val = np.zeros(design_matrix.shape[1])
        con_val[ix] = 1.0
        contrast = compute_contrast(labels, estimates, con_val)

        effect_map = contrast.effect[0]
        if roi_map is not None:
            effect_map = effect_map[roi_map.astype(bool)]
        effect_maps.append(effect_map)

    return np.corrcoef(*effect_maps)[0, 1]


def save_df(df, output_dir, subject, task, space, desc):
    """Saves a pandas DataFrame to a CSV file."""

    filename = f'sub-{subject}_task-{task}_space-{space}_desc-{desc}_df.csv'
    file = output_dir / filename
    df.to_csv(file, index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
