from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ants import apply_transforms
from ants.utils.convert_nibabel import from_nibabel
from bids import BIDSLayout, BIDSLayoutIndexer
from mvpa2.support.nibabel import afni_niml_dset
from nibabel import Nifti1Image, save
from nilearn.glm import compute_fixed_effects
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)
from nilearn.image import load_img, math_img
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map
from nilearn.surface import load_surf_data, vol_to_surf
from scipy.ndimage import binary_dilation
from surfplot import Plot
from xvfbwrapper import Xvfb


def compute_session_contrasts(layout, subject, task, space, fd_threshold,
                              hrf_model, smoothing_fwhm, contrast_defs):
    """Compute a dictionary and data frame of contrasts for each session."""

    contrast_dict = {label: {} for label in contrast_defs.keys()}

    sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
    for session in sessions:

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
        'run', 'events', subject=subject, session=session, scan_length=0.0)[0]
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
    variance_imgs = [imgs['effect_variance']
                     for imgs in contrast_dict.values()]
    psc_img, var_img, t_img = compute_fixed_effects(effect_imgs, variance_imgs)

    return psc_img, var_img, t_img


def plot_img_on_surf(img, layout, subject, roi_mask=None, views='lateral',
                     size=(3000, 1100), zoom=1.5, vmin=-1.0, vmax=1.0,
                     cmap_pos='YlOrRd_r', cmap_neg='YlGnBu'):
    """Plots a volume image on the inflated FreeSurfer surface."""

    inflated_files = sorted(
        layout.get('filename', subject=subject, suffix='inflated',
                   extension='surf.gii'))

    plot = Plot(*inflated_files, views=views, size=size, zoom=zoom)

    curv_files = sorted(
        layout.get('filename', subject=subject, extension='curv'))
    curv_map = np.concatenate([load_surf_data(f) for f in curv_files])
    curv_map_sign = np.sign(curv_map)
    _ = plot.add_layer(
        curv_map_sign, cmap='Greys', color_range=[-8.0, 4.0], cbar=False)

    # surf_map = volume_to_surface(img, layout, subject)

    # from nilearn.plotting.cm import cold_hot
    # _ = plot.add_layer(surf_map, cmap=cold_hot, color_range=[vmin, vmax],
    #                    zero_transparent=False, cbar=False)

    # surf_map_pos = np.where(surf_map > 0.0, surf_map, 0.0)
    # color_range_pos = [0.0, vmax]
    # _ = plot.add_layer(surf_map_pos, cmap=cmap_pos,
    #                    color_range=color_range_pos, zero_transparent=False,
    #                    cbar=False)

    # surf_map_neg = np.where(surf_map < 0.0, surf_map, 0.0)
    # color_range_neg = [vmin, 0.0]
    # _ = plot.add_layer(surf_map_neg, cmap=cmap_neg,
    #                    color_range=color_range_neg,
    #                    cbar=False)

    if roi_mask is not None:
        roi_map = volume_to_surface(roi_mask, layout, subject)
        roi_map_bin = np.where(roi_map > 0.0, 1.0, 0.0)
        # annot_files = layout.get('filename', subject=subject,
        #                          extension='.aparc.annot')
        # annot_map = np.concatenate([load_surf_data(f) for f in annot_files])
        # roi_maps = [np.where(annot_map == roi, 1.0, 0.0) for roi in rois]
        # roi_map = np.sum(roi_maps, axis=0)
        _ = plot.add_layer(roi_map_bin, cmap='brg', as_outline=True,
                           cbar=False)

    # roi_map_fs = np.concatenate(
    # [load_surf_data(f) for f in
    #  ['/Users/alexander/Research/slang/data/derivatives/code/roi_map_fs_lh.mgz',
    #   '/Users/alexander/Research/slang/data/derivatives/code/roi_map_fs_rh.mgz']])
    # roi_map_fs_bin = np.where(roi_map_fs > 0.0, 1.0, 0.0)

    # _ = plot.add_layer(roi_map_fs_bin, cmap='brg', as_outline=True, cbar=False)

    _ = plot.build()

    return plot.build()


def volume_to_surface(img, layout, subject):
    """Converts a volume image to a surface image."""

    pial_files = sorted(layout.get(
        'filename', subject=subject, extension='pial'))
    white_files = sorted(layout.get(
        'filename', subject=subject, extension='white'))
    maps = [vol_to_surf(img, surf_mesh=pial_file, inner_mesh=white_file)
            for pial_file, white_file in zip(pial_files, white_files)]

    return np.concatenate(maps)


def make_roi_mask(layout, subject, task, space, rois, hemi, dilation=None):
    """Creates a volumetric brain mask for a set of ROIs in the DKT atlas."""

    atlas_file = layout.get(
        'filename', subject=subject, task=task, space=space, desc='aparcaseg',
        suffix='dseg', extension='.nii.gz')[0]
    atlas_img = load_img(atlas_file)

    hemi_adjust = 1000 if hemi == 'L' else 2000
    hemi_rois = [roi + hemi_adjust for roi in rois]

    roi_mask = math_img('np.where(img == -99, 1, 0)', img=atlas_img)
    for roi in hemi_rois:  # Add volumetric rois to empty image one by one
        this_roi_mask = math_img(
            f'np.where(img == {roi}, 1, 0)', img=atlas_img)
        roi_mask = math_img('roi_mask + this_roi_mask', roi_mask=roi_mask,
                            this_roi_mask=this_roi_mask)

    if dilation is not None:
        old_data = roi_mask.get_fdata()
        new_data = binary_dilation(old_data, iterations=dilation)
        roi_mask = Nifti1Image(new_data, roi_mask.affine, roi_mask.header)

    return roi_mask


def make_roi_mask_glasser(ref_img, atlas_file, layout, subject, space, rois,
                          hemi, atlas_space='MNI152NLin2009cAsym',
                          dilation=None):
    """Uses ANTsPy to convert the Glasser atlas to subject space and selects ROIs."""

    atlas_img = load_img(atlas_file)

    ref_img_ants = from_nibabel(ref_img)
    atlas_img_ants = from_nibabel(atlas_img)

    mni_to_t1w_dict = {'from': atlas_space, 'to': 'T1w'}
    mni_to_t1w_transform = sorted(layout.get('filename', subject=subject,
                                             suffix='xfm', extension='.h5',
                                             **mni_to_t1w_dict))[0]
    transformlist = [mni_to_t1w_transform]
    if space != 'T1w':
        t1w_to_space_dict = {'from': 'T1w', 'to': space}
        t1w_to_space_transform = sorted(layout.get(
            'filename', subject=subject,
            suffix='xfm', extension='.h5', **
            t1w_to_space_dict))[0]
        transformlist.append(t1w_to_space_transform)

    atlas_img_t1w_ants = apply_transforms(ref_img_ants,
                                          atlas_img_ants,
                                          transformlist,
                                          interpolator='nearestNeighbor')
    atlas_img_t1w = atlas_img_t1w_ants.to_nibabel()

    hemi_adjust = 0 if hemi == 'L' else 1000
    hemi_rois = [roi + hemi_adjust for roi in rois]

    roi_mask = math_img('np.where(img == -99.0, 1.0, 0.0)', img=atlas_img_t1w)
    for roi in hemi_rois:  # Add volumetric rois to empty image one by one
        this_roi_mask = math_img(
            f'np.where(img == {roi}, 1.0, 0.0)', img=atlas_img_t1w)
        roi_mask = math_img('roi_mask + this_roi_mask', roi_mask=roi_mask,
                            this_roi_mask=this_roi_mask)

    if dilation is not None:
        old_data = roi_mask.get_fdata()
        new_data = binary_dilation(old_data, iterations=dilation)
        roi_mask = Nifti1Image(new_data, roi_mask.affine, roi_mask.header)

    return roi_mask


task = 'language'
space = 'T1w'
fd_threshold = 2.4
hrf_model = 'spm'
smoothing_fwhm = 5.0
contrast_defs = {
    'a-pseudo-minus-noise': (['audios_pseudo'],
                             ['audios_noise']),
    'v-pseudo-minus-noise': (['images_pseudo'],
                             ['images_noise'])}
froi_contrast_label = 'a-pseudo-minus-noise'
psc_contrast_label = 'v-pseudo-minus-noise'
rois = [129]
t_thresh = 2.0

derivatives_dir = Path(__file__).parent.parent
bids_dir = derivatives_dir.parent
fmriprep_dir = derivatives_dir / 'fmriprep'
freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
suma_dir = derivatives_dir / 'suma'
output_dir = derivatives_dir / 'pug'

atlas_file = str(derivatives_dir /
                 'fmriprep_no-submm/MNI_Glasser_HCP_v1.0.nii.gz')

output_dir.mkdir(exist_ok=True)

indexer = BIDSLayoutIndexer(force_index=str(freesurfer_dir))
layout = BIDSLayout(bids_dir, derivatives=[fmriprep_dir], indexer=indexer)

subjects = sorted(layout.get_subjects(desc='preproc'))
subjects = ['SA15']
psc_dfs = []
for subject in subjects:

    print(subject)

    anat_file = layout.get('filename', subject=subject, suffix='T1w',
                           extension='nii.gz')[0]

    contrast_dict = compute_session_contrasts(
        layout, subject, task, space, fd_threshold, hrf_model,
        smoothing_fwhm, contrast_defs)

    # aud_imgs = fixed_effects(contrast_dict['a-pseudo-minus-noise'])
    # vis_imgs = fixed_effects(contrast_dict['v-pseudo-minus-noise'])

    # aud_img = aud_imgs[2]
    # aud_img_thresh = math_img(f'np.where(img > {t_thresh}, -1.0, 0.0)',
    #                           img=aud_img)
    # vis_img = vis_imgs[2]
    # vis_img_thresh = math_img(f'np.where(img > {t_thresh}, -2.0, 0.0)',
    #                           img=vis_img)
    # overlap_img = math_img('np.where(img1 * img2 > 0.0, img3 * img4, 0.0)',
    #                        img1=aud_img_thresh, img2=vis_img_thresh,
    #                        img3=aud_img, img4=vis_img)
    # overlap_img_bin = math_img('np.where(img1 > 0.0, -3.0, 0.0)',
    #                        img1=overlap_img)

    # col_img = math_img('img1 + img2 + img3',
    #                    img1=aud_img_thresh, img2=vis_img_thresh,
    #                    img3=overlap_img_bin)

    # fig = plt.figure(figsize=(100.0, 20.0))
    # _ = plot_stat_map(col_img, anat_file, cut_coords=6, display_mode='z', cmap='Dark2', vmax=3.0, colorbar=False, figure=fig, annotate=False)

    # peak_voxel_aud = np.unravel_index(np.argmax(aud_img.get_fdata()), aud_img.shape)
    # peak_voxel_vis = np.unravel_index(np.argmax(vis_img.get_fdata()), aud_img.shape)

    # overlap_ix_z = 22
    # overlap_slice = overlap_img.get_fdata()[:, :, overlap_ix_z]
    # overlap_ix_xy = np.unravel_index(np.argmax(overlap_slice),
    #                                  overlap_slice.shape)
    # overlap_ix = (overlap_ix_xy[0], overlap_ix_xy[1], overlap_ix_z)

    # sessions = sorted(layout.get_sessions(subject=subject, desc='preproc'))
    # aud_pscs = []
    # aud_vars = []
    # vis_pscs = []
    # vis_vars = []
    # for session in sessions:
    #     aud_psc = contrast_dict['a-pseudo-minus-noise'][session]['effect_size'].get_fdata()[overlap_ix]
    #     aud_var = contrast_dict['a-pseudo-minus-noise'][session]['effect_variance'].get_fdata()[overlap_ix]
    #     aud_pscs.append(aud_psc)
    #     aud_vars.append(aud_var)
    #     vis_psc = contrast_dict['v-pseudo-minus-noise'][session]['effect_size'].get_fdata()[overlap_ix]
    #     vis_var = contrast_dict['v-pseudo-minus-noise'][session]['effect_variance'].get_fdata()[overlap_ix]
    #     vis_pscs.append(vis_psc)
    #     vis_vars.append(vis_var)
    # psc_df = pd.DataFrame({'subject': subject,
    #                          'session': sessions,
    #                          'aud_psc': aud_pscs,
    #                          'aud_var': aud_vars,
    #                          'vis_psc': vis_pscs,
    #                          'vis_var': vis_vars})

    psc_img, var_img, t_img = \
        fixed_effects(contrast_dict[froi_contrast_label])

    psc_img_thresh = \
        math_img(f'np.where(np.abs(t_img) > {t_thresh}, psc_img, 0.0)',
                 t_img=t_img, psc_img=psc_img)

    roi_masks = []
    for hemi in ['L', 'R']:
        roi_mask = make_roi_mask_glasser(t_img, atlas_file, layout,
                                         subject, space, rois, hemi,
                                         dilation=None)
        roi_masks.append(roi_mask)
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
    roi_mask = math_img('img1 + img2', img1=roi_masks[0], img2=roi_masks[1])

    # vdisplay = Xvfb()
    # vdisplay.start()
    fig = plot_img_on_surf(psc_img, layout, subject, roi_mask)
    _ = plt.tight_layout()
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / \
        f'sub-{subject}_task-{task}_space-{space}_desc-{froi_contrast_label}_plot.png'
    _ = plt.savefig(plot_file)
    # vdisplay.stop()

psc_df = pd.concat(psc_dfs)
psc_df_file = output_dir / \
    f'task-{task}_space-{space}_desc-{froi_contrast_label}_psc.csv'
psc_df.to_csv(psc_df_file, index=False, float_format='%.4f')
