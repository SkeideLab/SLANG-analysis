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


def compute_session_contrasts(layout, subject, task, space, fd_threshold,
                              hrf_model, contrast_defs):
    """Compute a dictionary and data frame of contrasts for each session."""

    aud_contrasts = {}
    vis_contrasts = {}

    sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
    for session in sessions:

        design_matrix, labels, estimates = compute_glm(layout, subject,
                                                       session, task, space,
                                                       fd_threshold, hrf_model,
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


def compute_glm(layout, subject, session, task, space, fd_threshold, hrf_model,
                use_single_trials=False):
    """Fits the first-level GLM on surface data for a single session."""

    event_cols = ['onset', 'duration', 'trial_type']
    events = layout.get_collections('run', 'events', subject=subject,
                                    session=session)[0]
    events = events.to_df()[event_cols]

    if use_single_trials:  # See https://nilearn.github.io/dev/auto_examples/07_advanced/plot_beta_series.html
        conditions = events['trial_type'].unique()
        condition_counter = {c: 0 for c in conditions}
        for trial_ix, trial in events.iterrows():
            condition = trial['trial_type']
            condition_counter[condition] += 1
            trial_name = f'{condition}_{condition_counter[condition]:02d}'
            events.loc[trial_ix, 'trial_type'] = trial_name

    surf_files = layout.get('filename', scope='derivatives', subject=subject,
                            session=session, task=task, space=space,
                            suffix='bold', extension='.func.gii')

    texture = np.concatenate([load_surf_data(f) for f in surf_files]).T

    n_scans = texture.shape[0]
    start_time = layout.get_metadata(surf_files[0])['StartTime']
    t_r = layout.get_metadata(surf_files[0])['RepetitionTime']
    frame_times = start_time + t_r * np.arange(n_scans)

    # Necessary to get confounds via *volumetric* fMRIPrep outputs
    # Due to https://github.com/nilearn/nilearn/issues/3479 and
    # https://github.com/nilearn/nilearn/blob/b1fa2e/nilearn/interfaces/fmriprep/load_confounds_utils.py#L19
    vol_file = layout.get('filename', scope='derivatives', subject=subject,
                          session=session, task=task, space='T1w',
                          desc='preproc', extension='.nii.gz')[0]
    confounds, sample_mask = load_confounds(
        vol_file, strategy=('motion', 'high_pass', 'wm_csf', 'scrub', 'compcor'),
        motion='basic', wm_csf='basic', scrub=0, fd_threshold=fd_threshold,
        std_dvars_threshold=None, compcor='anat_combined', n_compcor=6)

    design_matrix = make_first_level_design_matrix(frame_times, events,
                                                   hrf_model, drift_model=None,
                                                   add_regs=confounds)

    texture = texture[sample_mask]
    design_matrix = design_matrix.iloc[sample_mask]

    condition_cols = set(design_matrix.columns) - set(confounds.columns)
    for col in condition_cols:  # Scale task regressors to max of 1 for PSC
        scale_factor = 1.0 / max(design_matrix[col])
        design_matrix[col] = design_matrix[col] * scale_factor

    mean_texture = texture.mean(axis=0)
    texture = 100.0 * (texture / mean_texture - 1.0)

    labels, estimates = run_glm(texture, design_matrix.values)

    return design_matrix, labels, estimates


def compute_psc_contrast(labels, estimates, design_matrix,
                         conditions_plus, conditions_minus):
    """Extracts a contrast (incl. percent signal change) from the GLM."""

    contrast_values = np.zeros(design_matrix.shape[1])

    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)

    return compute_contrast(labels, estimates, contrast_values)


def contrast_to_df(contrast, label, subject, session):
    """Converts a Nilearn contrast object to data frame."""

    contrast_df = pd.DataFrame({
        'subject': f'sub-{subject}',
        'session': f'ses-{session}',
        'contrast': label,
        'effect': contrast.effect_size(),
        'variance': contrast.effect_variance(),
        'stat': contrast.stat()})

    contrast_df.insert(3, 'node', contrast_df.index)

    return contrast_df


def compute_fixed_effects(contrasts):
    """Computes fixed-effects model from a set of first level contrasts."""

    effects = [contrast.effect for contrast in contrasts]
    variances = [contrast.variance for contrast in contrasts]

    psc_maps, var_maps, t_maps = _compute_fixed_effects_params(
        effects, variances, precision_weighted=False)

    return psc_maps[0], var_maps[0], t_maps[0]


def surf_to_surf(derivatives_dataset, freesurfer_dir, source_subject,
                 target_subject, sval_filename, tval_filename=None):
    """Uses FreeSurfer to convert fsaverage surface annotations to fsnative."""

    freesurfer_dir = Path(freesurfer_dir)

    if Path(sval_filename).suffix == '.annot':
        sval_flag = ['--sval-annot', sval_filename]
    else:
        sval_flag = ['--sval', sval_filename]

    if tval_filename is None:
        tval_filename = sval_filename

    output_files = []
    for hemi in ['lh', 'rh']:

        input_file = freesurfer_dir / \
            f'{source_subject}/label/{hemi}.{sval_filename}'
        output_file = freesurfer_dir / \
            f'{target_subject}/label/{hemi}.{sval_filename}'

        if not output_file.exists():

            cmd = ['mri_surf2surf',
                   '--srcsubject', source_subject,
                   '--trgsubject', target_subject,
                   '--hemi', hemi,
                   *sval_flag,
                   '--tval', sval_filename,
                   '--sd', freesurfer_dir]
            cmd = ' '.join([str(elem) for elem in cmd])
            cmd = f'-c \'{cmd}\''  # See https://stackoverflow.com/a/62313159

            # # To use Docker instead of Singularity
            # from os import environ
            # environ['REPRONIM_USE_DOCKER'] = 'TRUE'

            derivatives_dataset.containers_run(
                cmd, container_name='code/containers/neurodesk-freesurfer',
                inputs=[str(input_file)], outputs=[str(output_file)],
                message=f'Convert surface annotations from {source_subject} to {subject}',
                explicit=True)

        else:
            print(f'Output surface file "{output_file}" exists, nothing to do')

        output_files.append(output_file)

    return output_files


def make_glasser_roi_map(derivatives_dir, freesurfer_dir, subject, roi_ixs):
    """Creates a ROI map for one or more Glasser ROIs in fsnative space."""

    derivatives_dataset = Dataset(derivatives_dir)
    atlas_files = surf_to_surf(derivatives_dataset, freesurfer_dir,
                               source_subject='fsaverage',
                               target_subject=f'sub-{subject}',
                               sval_filename='HCPMMP1.annot')

    atlas_map = np.concatenate([load_surf_data(f) for f in atlas_files])

    if isinstance(roi_ixs, (int, float, str)):
        roi_ixs = [int(roi_ixs)]
    else:
        roi_ixs = [int(ix) for ix in roi_ixs]

    return np.sum([atlas_map == roi_ix for roi_ix in roi_ixs], axis=0)


def make_surfplot(
        layout, subject, stat_map=None, roi_map_1=None, roi_map_2=None,
        add_curv=True, views='lateral', size=(1000, 300),
        zoom=2.0, cmap=cold_hot, cbar_label=None, vmin=-2.0, vmax=2.0):
    """Plots a statistical and/or ROI map(s) on the inflated FreeSurfer surface."""

    inflated_files = sorted(
        layout.get(
            'filename', subject=subject, suffix='inflated',
            extension='surf.gii'))
    plot = Plot(*inflated_files, views=views, size=size, zoom=zoom)

    if add_curv:
        curv_files = sorted(layout.get('filename', subject=subject,
                                       extension='curv'))
        curv_map = np.concatenate([load_surf_data(f) for f in curv_files])
        curv_map_sign = np.sign(curv_map)
        _ = plot.add_layer(curv_map_sign, cmap='Greys',
                           color_range=[-8.0, 4.0], cbar=False)

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


task = 'language'
space = 'fsnative'
fd_threshold = 2.4
hrf_model = 'spm'
contrast_defs = {'a-pseudo-minus-noise': (['audios_pseudo'], ['audios_noise']),
                 'v-pseudo-minus-noise': (['images_pseudo'], ['images_noise'])}
froi_contrast_label = 'a-pseudo-minus-noise'
psc_contrast_label = 'v-pseudo-minus-noise'
roi_ixs = 129
n_top_vertices = 500

derivatives_dir = Path(__file__).parent.parent
bids_dir = derivatives_dir.parent
fmriprep_dir = derivatives_dir / 'fmriprep'
freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
output_dir = derivatives_dir / 'nilearn'

indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir], indexer=indexer)

fsaverage_dir = Path(fetch_fsaverage(freesurfer_dir))
annot_file = fsaverage_dir / 'label/lh.HCPMMP1_combined.annot'

psc_dfs = []
for subject in ['SA15']:  # layout.get_subjects(desc='preproc'):

    print(f'\nPrecessing subject sub-{subject}\n')

    aud_contrasts, vis_contrasts = \
        compute_session_contrasts(layout, subject, task, space, fd_threshold,
                                  hrf_model, contrast_defs)

    psc_map, var_map, t_map = compute_fixed_effects(aud_contrasts.values())

    roi_map = make_glasser_roi_map(
        derivatives_dir, freesurfer_dir, subject, roi_ixs)

    t_map_roi = t_map * roi_map
    top_ixs = np.argsort(t_map_roi)[::-1][:n_top_vertices]
    froi_map = np.zeros_like(t_map_roi, dtype='int')
    froi_map[top_ixs] = 1

    _ = make_surfplot(layout, subject, t_map, roi_map, froi_map,
                      cbar_label='Spoken pseudowords\nminus noise ($t$)',
                      vmin=-11.0, vmax=11.0)

    output_sub_dir = output_dir / f'nilearn/sub-{subject}'
    output_sub_dir.mkdir(exist_ok=True, parents=True)
    plot_filename = f'sub-{subject}_task-{task}_space-{space}_desc-{froi_contrast_label}_plot.png'
    plot_file = output_sub_dir / plot_filename
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')

    for session, contrast in vis_contrasts.items():
        froi_psc = contrast.effect[0][froi_map].mean()
        psc_df = pd.DataFrame({'subject': subject,
                               'session': session,
                               'froi_psc': froi_psc},
                              index=[0])
        psc_dfs.append(psc_df)

output_group_dir = output_dir / 'nilearn/sub-group'
output_group_dir.mkdir(exist_ok=True, parents=True)
psc_df = pd.concat(psc_dfs)
psc_df_filename = f'sub-group_task-{task}_space-{space}_desc-{froi_contrast_label}_psc.csv'
psc_df_file = output_group_dir / psc_df_filename
psc_df.to_csv(psc_df_file, index=False, float_format='%.4f')
