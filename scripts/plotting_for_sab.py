from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer
from nilearn.glm.contrasts import (_compute_fixed_effects_params,
                                   compute_contrast)
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import load_surf_data

session = '01'
task = 'language'
space = 'fsnative'
fd_threshold = 2.4
hemi = 'L'
hrf_model = 'spm'

contrast_defs = {
    'a-noise-minus-null': (['audios_noise'], []),
    'a-pseudo-minus-null': (['audios_pseudo'], []),
    'a-words-minus-null': (['audios_words'], []),
    'v-noise-minus-null': (['images_noise'], []),
    'v-pseudo-minus-null': (['images_pseudo'], []),
    'v-words-minus-null': (['images_words'], [])}

rois = [7, 9, 30, 34]
t_threshold = 3.0
t_vmax = 8.0
psc_vmax = 2.0

bids_dir = Path('/ptmp/aenge/slang/data')
fmriprep_dir = bids_dir / 'derivatives/fmriprep'
freesurfer_dir = bids_dir / 'derivatives/fmriprep/sourcedata/freesurfer'
output_dir = bids_dir / 'derivatives/nilearn_plots_for_sab'

indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir, indexer=indexer)

for subject in ['SA15']:  # sorted(layout.get_subjects()):

    contrast_dict = {label: [] for label in contrast_defs.keys()}
    contrast_dfs = []

    output_subdir = output_dir / f'sub-{subject}'
    output_subdir.mkdir(parents=True, exist_ok=True)

    for session in sorted(layout.get_sessions()):

        event_cols = ['onset', 'duration', 'trial_type']
        events = layout.get_collections('run', 'events', subject=subject,
                                        session=session)[0].to_df()[event_cols]

        surf_files = layout.get(
            'filename', scope='derivatives', subject=subject, session=session,
            task=task, space=space, hemi=hemi, extension='.func.gii')

        texture = np.concatenate([load_surf_data(f).T for f in surf_files])

        n_scans = texture.shape[0]
        start_time = layout.get_metadata(surf_files[0])['StartTime']
        t_r = layout.get_metadata(surf_files[0])['RepetitionTime']
        frame_times = start_time + t_r * np.arange(n_scans)

        # Necessary to get confounds via *volumetric* fMRIPrep outputs
        # Due to https://github.com/nilearn/nilearn/issues/3479 and
        # https://github.com/nilearn/nilearn/blob/b1fa2e/nilearn/interfaces/fmriprep/load_confounds_utils.py#L19
        vol_file = layout.get(
            'filename', scope='derivatives', subject=subject, session=session,
            task=task, space='T1w', desc='preproc', extension='.nii.gz')[0]
        confounds, sample_mask = load_confounds(
            vol_file,
            strategy=('motion', 'high_pass', 'wm_csf', 'scrub', 'compcor'),
            motion='basic', wm_csf='basic',
            scrub=0, fd_threshold=fd_threshold, std_dvars_threshold=None,
            compcor='anat_combined', n_compcor=6)

        design_matrix = make_first_level_design_matrix(
            frame_times, events, hrf_model, drift_model=None,
            add_regs=confounds)

        texture = texture[sample_mask]
        design_matrix = design_matrix.iloc[sample_mask]

        condition_cols = set(design_matrix.columns) - set(confounds.columns)
        for col in condition_cols:  # Scale task regressors to max of 1 for PSC
            scale_factor = 1.0 / max(design_matrix[col])
            design_matrix[col] = design_matrix[col] * scale_factor

        mean_texture = texture.mean(axis=0)
        texture = 100.0 * (texture / mean_texture - 1.0)

        labels, estimates = \
            run_glm(texture, design_matrix.values)

        for label, condition_tuple in contrast_defs.items():

            contrast_values = np.zeros(design_matrix.shape[1])
            conditions_plus = condition_tuple[0]
            conditions_minus = condition_tuple[1]
            for col_ix, column in enumerate(design_matrix.columns):
                if column in conditions_plus:
                    contrast_values[col_ix] = 1.0 / len(conditions_plus)
                if column in conditions_minus:
                    contrast_values[col_ix] = -1.0 / len(conditions_minus)

            contrast = compute_contrast(
                labels, estimates, contrast_values, contrast_type='t')
            contrast_dict[label].append(contrast)

            contrast_df = pd.DataFrame({'subject': f'sub-{subject}',
                                        'session': f'ses-{session}',
                                        'contrast': label,
                                        'effect': contrast.effect_size(),
                                        'variance': contrast.effect_variance(),
                                        'stat': contrast.stat()})
            contrast_df.insert(3, 'node', contrast_df.index)
            contrast_dfs.append(contrast_df)

    hemi_fs = hemi.lower() + 'h'
    inflated_file = layout.get('filename', subject=subject, suffix=hemi_fs,
                               extension='.inflated')[0]
    sulc_file = layout.get('filename', subject=subject, suffix=hemi_fs,
                           extension='.sulc')[0]

    annot_file = Path(
        inflated_file).parent.parent / f'label/{hemi_fs}.aparc.annot'
    annot_data = load_surf_data(annot_file)

    peak_nodes = {}

    for label, contrasts in contrast_dict.items():

        effects = [contrast.effect for contrast in contrasts]
        variances = [contrast.variance for contrast in contrasts]
        fixed_fx_effect, fixed_fx_variance, fixed_fx_stat = \
            _compute_fixed_effects_params(effects, variances,
                                          precision_weighted=False)

        t_map = fixed_fx_stat[0]
        t_map_pos = np.where(t_map > 0.0, t_map, 0.0)

        roi_mask = np.isin(annot_data, list(rois))
        t_map_masked = t_map_pos * roi_mask
        peak_map = np.where(t_map_masked == t_map_masked.max(), 1.0, 0.0)

        psc_map = fixed_fx_effect[0]
        psc_map_thresh = np.where(t_map > t_threshold, fixed_fx_effect, 0.0)
        psc_map_masked = psc_map_thresh * roi_mask

        hemi_plot = 'left' if hemi == 'L' else 'right'
        for view in ['lateral', 'ventral']:

            _ = plot_surf_stat_map(
                inflated_file, t_map_pos, sulc_file, hemi_plot, view,
                threshold=t_threshold, vmax=t_vmax)
            t_map_pos_file = output_subdir / \
                f'sub-{subject}_desc-{label}-{view}_tstat.png'
            _ = plt.savefig(t_map_pos_file, dpi=300)
            _ = plt.close()

            _ = plot_surf_stat_map(
                inflated_file, t_map_masked, sulc_file, hemi_plot, view,
                threshold=t_threshold, vmax=t_vmax)
            t_map_masked_file = output_subdir / \
                f'sub-{subject}_desc-{label}-{view}-masked_tstat.png'
            _ = plt.savefig(t_map_masked_file, dpi=300)
            _ = plt.close()

            _ = plot_surf_stat_map(
                inflated_file, peak_map, sulc_file, hemi_plot, view, vmax=1.0)
            peak_map_file = output_subdir / \
                f'sub-{subject}_desc-{label}-{view}_peak.png'
            _ = plt.savefig(peak_map_file, dpi=300)
            _ = plt.close()

            _ = plot_surf_stat_map(
                inflated_file, psc_map_thresh, sulc_file, hemi_plot, view,
                threshold=0.0001, vmax=psc_vmax)
            psc_map_thresh_file = output_subdir / \
                f'sub-{subject}_desc-{label}-{view}_psc.png'
            _ = plt.savefig(psc_map_thresh_file, dpi=300)
            _ = plt.close()

            _ = plot_surf_stat_map(
                inflated_file, psc_map_masked, sulc_file, hemi_plot, view,
                threshold=0.0001, vmax=psc_vmax)
            psc_map_masked_file = output_subdir / \
                f'sub-{subject}_desc-{label}-{view}-masked_psc.png'
            _ = plt.savefig(psc_map_masked_file, dpi=300)
            _ = plt.close()

        peak_node = np.nonzero(peak_map)[0][0]
        peak_nodes[label] = peak_node

        fixed_df = pd.DataFrame({'subject': f'sub-{subject}',
                                 'session': f'fixed_effect',
                                 'contrast': label,
                                 'effect': fixed_fx_effect[0],
                                 'variance': fixed_fx_variance[0],
                                 'stat': fixed_fx_stat[0]})
        fixed_df.insert(3, 'node', fixed_df.index)
        contrast_dfs.append(fixed_df)

    contrast_df = pd.concat(contrast_dfs)
    contrast_df_file = output_subdir / f'sub-{subject}_contrasts.csv'
    # contrast_df.to_csv(contrast_df_file, index=False, float_format='%.3f')

    peak_dfs = []
    for label, peak_node in peak_nodes.items():
        peak_df = contrast_df[contrast_df['node'] == peak_node]
        peak_df.insert(2, 'peak_contrast', label)
        peak_dfs.append(peak_df)

    peak_df = pd.concat(peak_dfs)
    peak_df_file = output_subdir / f'sub-{subject}_peaks.csv'
    peak_df.to_csv(peak_df_file, index=False, float_format='%.3f')

scan_df = layout.get_collections('session',
                                 types='scans',
                                 variables='acq_time',
                                 merge=True,
                                 datatype='func').to_df()
scan_df['acq_date'] = pd.to_datetime(scan_df['acq_time']).dt.date
scan_df = scan_df[['subject', 'session', 'acq_date']]
scan_df.to_csv(output_dir / 'acq_dates.csv', index=False)
