from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bids import BIDSLayout, BIDSLayoutIndexer
from nilearn.glm import compute_fixed_effects
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)
from nilearn.image import load_img, math_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMasker
from nilearn.surface import load_surf_data, vol_to_surf
from surfplot import Plot


def main():
    """Plots a functional region of interest and extracts PSC for each session."""

    task = 'language'
    space = 'T1w'
    fd_threshold = 2.4
    hrf_model = 'spm'
    smoothing_fwhm = 4.0
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
    output_dir = derivatives_dir / 'nilearn_vol'

    indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
    layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir, indexer=indexer)

    psc_dfs = []
    for subject in sorted(layout.get_subjects()):

        contrast_dict = compute_session_contrasts(
            layout, subject, task, space, fd_threshold, hrf_model, 
            smoothing_fwhm, contrast_defs)

        psc_img, var_img, t_img = \
            fixed_effects(contrast_dict[froi_contrast_label])

        psc_img_thresh = \
            math_img(f'np.where(np.abs(t_img) > {t_thresh}, psc_img, 0.0)',
                     t_img=t_img, psc_img=psc_img)

        fig = plot_img_on_surf(psc_img_thresh, layout, subject, rois)
        _ = plt.tight_layout()
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_file = plot_dir / \
            f'sub-{subject}_task-{task}_space-{space}_desc-{froi_contrast_label}_plot.png'
        _ = plt.savefig(plot_file)

        for hemi in ['L', 'R']:
            roi_mask = make_roi_mask(layout, subject, task, space, rois, hemi)
            froi_mask = math_img(
                f'roi_mask * np.where(t_img > {t_thresh}, 1.0, 0.0)',
                roi_mask=roi_mask, t_img=t_img)
            froi_masker = NiftiMasker(froi_mask)
            froi_size = froi_mask.get_fdata().sum(dtype=np.int64)
            psc_contrasts = contrast_dict[psc_contrast_label]
            for session, contrast_imgs in psc_contrasts.items():
                psc_session_img = contrast_imgs['effect_size']
                froi_pscs = froi_masker.fit_transform(psc_session_img)
                t_session_img = contrast_imgs['stat']
                froi_ts = froi_masker.fit_transform(t_session_img)
                psc_df = pd.DataFrame({'subject': subject,
                                       'session': session,
                                       'hemi': hemi,
                                       'froi_size': froi_size,
                                       'froi_psc': froi_pscs.mean(),
                                       'froi_std': froi_pscs.std(),
                                       'froi_stat': froi_ts.mean()},
                                       index=[0])
                psc_dfs.append(psc_df)

    psc_df = pd.concat(psc_dfs)
    psc_df_file = output_dir / \
        f'task-{task}_space-{space}_desc-{froi_contrast_label}_psc.csv'
    psc_df.to_csv(psc_df_file, index=False, float_format='%.4f')


def compute_session_contrasts(layout, subject, task, space, fd_threshold, 
                              hrf_model, smoothing_fwhm, contrast_defs):
    """Compute a dictionary and data frame of contrasts for each session."""
    
    contrast_dict = {label: {} for label in contrast_defs.keys()}

    for session in sorted(layout.get_sessions(subject=subject)):

        first_level_model = compute_glm(
            layout, subject, session, task, space, fd_threshold, hrf_model,
            smoothing_fwhm)

        for label, condition_tuple in contrast_defs.items():

            conditions_plus = condition_tuple[0]
            conditions_minus = condition_tuple[1]
            contrast_imgs = compute_psc_contrast(
                first_level_model, conditions_plus, conditions_minus)
            contrast_dict[label][session] = contrast_imgs

    return contrast_dict


def compute_glm(layout, subject, session, task, space, fd_threshold,
                hrf_model, smoothing_fwhm):
    """Fits the first-level GLM on volume data for a single session."""

    func_file = layout.get(
        'filename', scope='derivatives', subject=subject, session=session,
        task=task, space=space, desc='preproc', extension='.nii.gz')[0]
    func_img = load_img(func_file)
    n_scans = func_img.shape[-1]

    event_cols = ['onset', 'duration', 'trial_type']
    events = layout.get_collections(
        'run', 'events', subject=subject, session=session)[0]
    events = events.to_df()[event_cols]

    design_matrix, sample_mask = make_design_matrix(
        layout, events, func_file, n_scans, fd_threshold, hrf_model)

    first_level_model = FirstLevelModel(
        slice_time_ref=0.5, hrf_model=hrf_model, drift_model=None,
        smoothing_fwhm=smoothing_fwhm, signal_scaling=0)

    first_level_model.fit(func_img,
                          design_matrices=design_matrix,
                          sample_masks=sample_mask)

    return first_level_model


def make_design_matrix(layout, events, func_file, n_scans, fd_threshold,
                       hrf_model):
    """Creates a design matrix with task regressors scaled to 1.0 for PSC."""

    start_time = layout.get_metadata(func_file)['StartTime']
    t_r = layout.get_metadata(func_file)['RepetitionTime']
    frame_times = start_time + t_r * np.arange(n_scans)

    confounds, sample_mask = load_confounds(
        func_file,
        strategy=('motion', 'high_pass', 'wm_csf', 'scrub', 'compcor'),
        motion='basic', wm_csf='basic',
        scrub=0, fd_threshold=fd_threshold, std_dvars_threshold=None,
        compcor='anat_combined', n_compcor=6)

    design_matrix = make_first_level_design_matrix(
        frame_times, events, hrf_model, drift_model=None,
        add_regs=confounds)

    condition_cols = set(design_matrix.columns) - set(confounds.columns)
    for col in condition_cols:  # Scale task regressors to max of 1 for PSC
        scale_factor = 1.0 / max(design_matrix[col])
        design_matrix[col] = design_matrix[col] * scale_factor

    return design_matrix, sample_mask


def compute_psc_contrast(first_level_model, conditions_plus, conditions_minus):
    """Extracts a contrast (incl. percent signal change) from the GLM."""

    design_matrix = first_level_model.design_matrices_[0]
    contrast_values = np.zeros(design_matrix.shape[1])

    for col_ix, column in enumerate(design_matrix.columns):
        if column in conditions_plus:
            contrast_values[col_ix] = 1.0 / len(conditions_plus)
        if column in conditions_minus:
            contrast_values[col_ix] = -1.0 / len(conditions_minus)

    return first_level_model.compute_contrast(
        contrast_values, output_type='all')


def fixed_effects(contrast_dict):
    """Computes fixed effects for a dictionary of contrast images."""

    effect_imgs = [imgs['effect_size'] for imgs in contrast_dict.values()]
    variance_imgs = [imgs['effect_variance'] for imgs in contrast_dict.values()]
    psc_img, var_img, t_img = compute_fixed_effects(effect_imgs, variance_imgs)

    return psc_img, var_img, t_img


def plot_img_on_surf(img, layout, subject, rois=None, views='lateral',
                     size=(3000, 1100), zoom=1.7, vmin=-1.0, vmax=1.0,
                     cmap_pos='YlOrRd_r', cmap_neg='YlGnBu'):
    """Plots a volume image on the inflated FreeSurfer surface."""

    inflated_files = layout.get('filename', subject=subject, session=None,
                                suffix='inflated', extension='.surf.gii')
    plot = Plot(*inflated_files, views=views, size=size, zoom=zoom)

    curv_files = layout.get('filename', subject=subject, extension='.curv')
    curv_map = np.concatenate([load_surf_data(f) for f in curv_files])
    curv_map_sign = np.sign(curv_map)
    _ = plot.add_layer(
        curv_map_sign, cmap='Greys', color_range=[-8.0, 4.0], cbar=False)

    surf_map = volume_to_surface(img, layout, subject)

    surf_map_pos = np.where(surf_map > 0.0, surf_map, 0.0)
    color_range_pos = [0.0, vmax]
    _ = plot.add_layer(surf_map_pos, cmap=cmap_pos, color_range=color_range_pos)

    surf_map_neg = np.where(surf_map < 0.0, surf_map, 0.0)
    color_range_neg = [vmin, 0.0]
    _ = plot.add_layer(surf_map_neg, cmap=cmap_neg, color_range=color_range_neg)

    if rois is not None:
        annot_files = layout.get('filename', subject=subject,
                                extension='.aparc.annot')
        annot_map = np.concatenate([load_surf_data(f) for f in annot_files])
        roi_maps = [np.where(annot_map == roi, 1.0, 0.0) for roi in rois]
        roi_map = np.sum(roi_maps, axis=0)
        _ = plot.add_layer(roi_map, cmap='brg', as_outline=True, cbar=False)
    
    return plot.build()


def volume_to_surface(img, layout, subject):
    """Converts a volume image to a surface image."""

    pial_files = layout.get('filename', subject=subject, extension='.pial')
    white_files = layout.get('filename', subject=subject, extension='.white')

    maps = [vol_to_surf(img, surf_mesh=pial_file, inner_mesh=white_file,
                        interpolation='nearest')
            for pial_file, white_file in zip(pial_files, white_files)]

    return np.concatenate(maps)


def make_roi_mask(layout, subject, task, space, rois, hemi):
    """Creates a volumetric brain mask for a set of ROIs in the DKT atlas."""

    atlas_file = layout.get(
        'filename', subject=subject, task=task, space=space, desc='aparcaseg',
        suffix='dseg', extension='.nii.gz')[0]
    atlas_img = load_img(atlas_file)

    hemi_adjust = 1000 if hemi == 'L' else 2000
    hemi_rois = [roi + hemi_adjust for roi in rois]

    roi_mask = math_img('np.where(img == -99, 1, 0)', img=atlas_img)
    for roi in hemi_rois: # Add volumetric rois to empty image one by one
        this_roi_mask = math_img(f'np.where(img == {roi}, 1, 0)', img=atlas_img)
        roi_mask = math_img('roi_mask + this_roi_mask', roi_mask=roi_mask, 
                            this_roi_mask=this_roi_mask)

    return roi_mask


if __name__ == "__main__":
    main()
