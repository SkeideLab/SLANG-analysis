from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer
from nilearn.glm.contrasts import (_compute_fixed_effects_params,
                                   compute_contrast)
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.surface import load_surf_data
from surfplot import Plot
import matplotlib.pyplot as plt

task = 'language'
space = 'fsnative'
fd_threshold = 2.4
hrf_model = 'spm'
contrast_defs = {'a-pseudo-minus-noise': (['audios_pseudo'], ['audios_noise']),
                 'v-pseudo-minus-noise': (['images_pseudo'], ['images_noise'])}
froi_contrast_label = 'a-pseudo-minus-noise'
psc_contrast_label = 'v-pseudo-minus-noise'
rois = [30, 34]
t_thresh = 2.0

derivatives_dir = Path(__file__).parent.parent
bids_dir = derivatives_dir.parent
fmriprep_dir = derivatives_dir / 'fmriprep'
freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
output_dir = derivatives_dir / 'nilearn'

indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir, indexer=indexer)


def compute_session_contrasts(layout, subject, task, space, hemi,
                              fd_threshold, hrf_model, contrast_defs):
    """Compute a dictionary and data frame of contrasts for each session."""
    
    contrast_dict = {label: {} for label in contrast_defs.keys()}
    contrast_dfs = []

    for session in ['01', '02', '03', '04']: # sorted(layout.get_sessions(subject=subject)):

        design_matrix, labels, estimates = compute_glm(
            layout, subject, session, task, space, hemi, fd_threshold,
            hrf_model)

        for label, condition_tuple in contrast_defs.items():

            conditions_plus = condition_tuple[0]
            conditions_minus = condition_tuple[1]
            contrast = compute_psc_contrast(labels, estimates, design_matrix,
                                            conditions_plus, conditions_minus)
            contrast_dict[label][session] = contrast

            contrast_df = contrast_to_df(contrast, label, subject, session)
            contrast_dfs.append(contrast_df)

    contrast_df = pd.concat(contrast_dfs)

    return contrast_dict, contrast_df


def compute_glm(layout, subject, session, task, space, hemi, fd_threshold,
                hrf_model):
    """Fits the first-level GLM on surface data for a single session."""

    event_cols = ['onset', 'duration', 'trial_type']
    events = layout.get_collections(
        'run', 'events', subject=subject, session=session)[0]
    events = events.to_df()[event_cols]

    surf_files = layout.get(
        'filename', scope='derivatives', subject=subject,
        session=session, task=task, space=space, hemi=hemi,
        extension='.func.gii')

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

    contrast = compute_contrast(
        labels, estimates, contrast_values, contrast_type='t')
    
    return contrast


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



psc_dfs = []
for subject in ['SA27']: # get_subjects():

    for hemi in ['L']: # ['L', 'R']:

        inflated_file = layout.get('filename', subject=subject, hemi=hemi,
                                   suffix='inflated', extension='.surf.gii')[0]

        hemi_fs = f'{hemi.lower()}h'
        curv_file = layout.get('filename', subject=subject, suffix=hemi_fs,
                               extension='.curv')[0]
        curv_map = load_surf_data(curv_file)
        curv_map_sign = np.sign(curv_map)

        annot_file = layout.get('filename', subject=subject, suffix=hemi_fs,
                                extension='.aparc.annot')[0]
        annot_data = load_surf_data(annot_file)
        roi_maps = [np.where(annot_data == roi, 1.0, 0.0) for roi in rois]
        roi_map = np.sum(roi_maps, axis=0)

        contrast_dict, contrast_df = compute_session_contrasts(
            layout, subject, task, space, hemi, fd_threshold, hrf_model,
            contrast_defs)

        psc_map, var_map, t_map = \
            compute_fixed_effects(contrast_dict[froi_contrast_label].values())

        psc_map_pos = np.where(t_map > t_thresh, psc_map, 0.0)
        psc_map_neg = np.where(t_map < -t_thresh, psc_map, 0.0)
        psc_map_thresh = np.where(np.abs(t_map) > t_thresh, psc_map, 0.0)

        plot = Plot(inflated_file, views = 'lateral', size=(2000, 1400), zoom=1.7)
        _ = plot.add_layer(curv_map_sign, cmap='Greys',
                            color_range=[-8.0, 4.0], cbar=False)
        _ = plot.add_layer(psc_map_pos, cmap='YlOrRd_r',
                           color_range=[0.0, 1.0], cbar=False)
        _ = plot.add_layer(psc_map_neg, cmap='YlGnBu',
                           color_range=[-1.0, 0.0], cbar=False)
        _ = plot.add_layer(roi_map, cmap='brg', as_outline=True, cbar=False)
        fig = plot.build()

        _ = plt.tight_layout()
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_dir / \
            f'sub-{subject}_task-{task}_space-{space}_hemi-{hemi}_desc-{froi_contrast_label}_plot.png'
        _ = plt.savefig(plot_file)

        froi_mask = roi_map * np.where(t_map > t_thresh, 1.0, 0.0)
        froi_mask = froi_mask.astype('bool')
        
        psc_contrasts = contrast_dict[psc_contrast_label]
        for session, contrast in psc_contrasts.items():
            froi_psc = contrast.effect[0][froi_mask].mean()
            psc_df = pd.DataFrame({'subject': subject,
                                   'session': session,
                                   'hemi': hemi,
                                   'froi_psc': froi_psc},
                                   index=[0])
            psc_dfs.append(psc_df)

psc_df = pd.concat(psc_dfs)
psc_df_file = output_dir / \
    f'task-{task}_space-{space}_hemi-{hemi}_desc-{froi_contrast_label}_psc.csv'
psc_df.to_csv(psc_df_file, index=False, float_format='%.4f')






        # t_map_thresh = np.where(np.abs(t_map) > t_thresh, t_map, 0.0)

        # roi_mask = np.isin(annot_data, list(rois))
        # t_map_masked = t_map_pos * roi_mask
        # peak_map = np.where(t_map_masked == t_map_masked.max(), 1.0, 0.0)

        # psc_map = fixed_fx_effect[0]
        # psc_map_thresh = np.where(np.abs(t_map) > t_thresh, psc_map, 0.0)
        # psc_map_pos = np.where(t_map > t_thresh, psc_map, 0.0)
        # psc_map_neg = np.where(t_map < -t_thresh, psc_map, 0.0)

        # from nilearn.plotting import plot_surf_stat_map
        # import matplotlib.pyplot as plt
        # inflated_file = '/Users/alexander/Research/slang/data/derivatives/fmriprep/sourcedata/freesurfer/sub-SA27/surf/lh.inflated'
        # curv_file = '/Users/alexander/Research/slang/data/derivatives/fmriprep/sourcedata/freesurfer/sub-SA27/surf/lh.curv'
        # curv_data = load_surf_data(curv_file)
        # curv_data_sign = np.sign(curv_data) / 6 + 0.8
        # _ = plot_surf_stat_map(inflated_file, t_map, bg_map=curv_data_sign, threshold=3.0, vmax=12.0)
        # plt.savefig('nilearn.png', dpi=600)

        # from nilearn.plotting import plot_surf
        # inflated_file = '/Users/alexander/Research/slang/data/derivatives/freesurfer/sub-SA27/surf/lh.inflated'
        # curv_file = '/Users/alexander/Research/slang/data/derivatives/freesurfer/sub-SA27/surf/lh.curv'
        # curv_map = load_surf_data(curv_file)
        # curv_map_sign = (np.sign(curv_map) + 1.0) / 20.0 + 0.5
        # sulc_file = '/Users/alexander/Research/slang/data/derivatives/freesurfer/sub-SA27/surf/lh.sulc'
        # sulc_map = load_surf_data(sulc_file)
        # sulc_map_norm = sulc_map / sulc_map.max()
        # bg_map = curv_map_sign + sulc_map_norm / 6.0
        # zero_map = np.zeros_like(bg_map)
        # _ = plot_surf(inflated_file, zero_map, bg_map, darkness=1.0)
        # plt.savefig('nilearn_anat.png', dpi=1200)

        # set_3d_backend('notebook')
        # subject_fs = f'sub-{subject}'
        # hemi_fs = f'{hemi.lower()}h'
        # brain = Brain(subject=subject_fs,
        #             hemi=hemi_fs,
        #             surf='inflated',
        #             cortex=[(0.5, 0.5, 0.5), (0.3, 0.3, 0.3)],
        #             size=(3000, 2500),
        #             background='white',
        #             subjects_dir=freesurfer_dir,
        #             offscreen=True)
        # brain.add_data(t_map, fmin=3.0, fmid=3.0000001, fmax=12.0, transparent=True)
        # brain.save_image('test.png')
        # brain.close()

        # from surfplot import Plot
        # hemi_lh = '/Users/alexander/Research/slang/data/derivatives/fmriprep/sub-SA27/anat/sub-SA27_run-1_hemi-L_inflated.surf.gii'
        # curv_file = '/Users/alexander/Research/slang/data/derivatives/fmriprep/sourcedata/freesurfer/sub-SA27/surf/lh.curv'
        # # hemi_lh = '/Users/alexander/Research/slang/data/derivatives/templateflow/tpl-subSA27/tpl-subSA27_hemi-L_den-164k_inflated.surf.gii'
        # # curv_file = '/Users/alexander/Research/slang/data/derivatives/freesurfer/sub-SA27/surf/lh.curv'
        # curv_map = load_surf_data(curv_file)
        # curv_map_sign = np.sign(curv_map)
        # plot = Plot(hemi_lh, views = 'lateral', size=(2000, 1500))
        # _ = plot.add_layer(curv_map_sign, cmap='Greys',
        #                     color_range=[-4.0, 4.0], cbar=False)
        # # _ = plot.add_layer(psc_map, cmap='cold_hot', color_range=[-2.0, 2.0], alpha=0.6)
        # _ = plot.add_layer(psc_map_pos, cmap='YlOrRd_r', color_range=[0.0, 2.0])
        # _ = plot.add_layer(psc_map_neg, cmap='YlGnBu', color_range=[-2.0, 0.0])
        # fig = plot.build()
        # plt.savefig('surfplot.png', dpi=300)
