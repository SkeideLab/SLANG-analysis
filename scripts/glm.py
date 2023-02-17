from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bids import BIDSLayout
from nibabel import gifti, save
from nilearn.glm import fdr_threshold
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.surface import load_surf_data
from surfplot import Plot
from xvfbwrapper import Xvfb


def save_gifti(img, path):
    """Save NumPy array as surface GIfTI file."""

    gifti_array = gifti.GiftiDataArray(img)
    gifti_img = gifti.GiftiImage(darrays=[gifti_array])
    save(gifti_img, path)


# User inputs
session = '01'
task = 'language'
space = 'fsnative'
slice_start_time = 1.176
hrf_model = 'glover + derivative'
contrast_pos = ['audios_noise', 'audios_pseudo', 'audios_words']
contrast_neg = ['images_noise', 'images_pseudo', 'images_words']
contrast_name = 'audios-minus-images'
vmax = 12

# Define directories
bids_dir = Path('/ptmp/aenge/slang/data')
fmriprep_dir = bids_dir / 'derivatives/fmriprep'
freesurfer_dir = bids_dir / 'derivatives/fmriprep/sourcedata/freesurfer'
output_dir = bids_dir / 'derivatives/nilearn'

# Read BIDS layout
layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir)

# Loop over participants and sessions
fig_dict = {}
participant_dirs = sorted(fmriprep_dir.glob('sub-*/'))
participant_dirs = [d for d in participant_dirs if d.is_dir()]
for participant_ix, participant_dir in enumerate(participant_dirs):
    sub = participant_dir.name
    fig_dict[sub] = {}
    session_dirs = sorted(participant_dir.glob('ses-*'))
    for session_ix, session_dir in enumerate(session_dirs):
        ses = session_dir.name

        # Read events
        events_file = bids_dir / sub / ses / 'func' / \
            f'{sub}_{ses}_task-{task}_run-1_events.tsv'
        events = pd.read_csv(events_file, sep='\t')

        # Load nuisance regressors
        func_dir = fmriprep_dir / sub / ses / 'func'
        nuisance_file = func_dir / \
            f'{sub}_{ses}_task-{task}_run-1_desc-confounds_timeseries.tsv'
        all_nuisance_regs = pd.read_csv(nuisance_file, sep='\t')
        nuisance_cols = ['trans_x', 'trans_y', 'trans_z',
                         'rot_x', 'rot_y', 'rot_z']
        nuisance_regs = all_nuisance_regs[nuisance_cols]

        # Create output directory
        output_sub_dir = output_dir / sub / ses / 'surf'
        output_sub_dir.mkdir(parents=True, exist_ok=True)

        # Load surfacedata
        left_file = func_dir / \
            f'{sub}_{ses}_task-{task}_run-1_hemi-L_space-{space}_bold.func.gii'
        right_file = func_dir / \
            f'{sub}_{ses}_task-{task}_run-1_hemi-R_space-{space}_bold.func.gii'
        left_data = load_surf_data(str(left_file))
        right_data = load_surf_data(str(right_file))
        data = np.concatenate([left_data, right_data])

        # Construct design matrix
        t_r = layout.get_tr()
        n_scans = data.shape[1]
        frame_times = t_r * np.arange(n_scans) + slice_start_time
        design_matrix = make_first_level_design_matrix(
            frame_times, events, hrf_model, add_regs=nuisance_regs)
        design_matrix_file = output_sub_dir / 'design_matrix.tsv'
        design_matrix.to_csv(design_matrix_file, sep='\t')

        # Fit the GLM
        labels, estimates = run_glm(data.T, design_matrix.values)

        # Save beta maps
        conditions = sorted(np.unique(events['trial_type']))
        for condition in conditions:
            contrast_vector = (design_matrix.columns == condition) * 1.0
            contrast = compute_contrast(
                labels, estimates, contrast_vector, contrast_type='t')
            effect_map = contrast.effect
            gifti_file = output_sub_dir / f'{condition}_beta.gii'
            save_gifti(effect_map, gifti_file)

        # Compute contrast
        contrast_vector = \
            sum([design_matrix.columns == x for x in contrast_pos]) - \
            sum([design_matrix.columns == x for x in contrast_neg])
        contrast = compute_contrast(
            labels, estimates, contrast_vector, contrast_type='t')

        # Save contrast map
        z_map = contrast.z_score()
        gifti_file = output_sub_dir / f'{contrast_name}_z.gii'
        save_gifti(z_map, gifti_file)

        # Threshold contrast map
        threshold = 3.1  # fdr_threshold(z_map, alpha=0.05)
        z_map_thresh_pos = (z_map > threshold) * z_map
        z_map_thresh_neg = (z_map < -threshold) * -z_map

        # Load anatomical data
        anat_dir = fmriprep_dir / sub / 'anat'
        lh_file = anat_dir / f'{sub}_run-1_hemi-L_inflated.surf.gii'
        rh_file = anat_dir / f'{sub}_run-1_hemi-R_inflated.surf.gii'
        sulc_files = [freesurfer_dir / f'{sub}/surf/lh.sulc',
                      freesurfer_dir / f'{sub}/surf/rh.sulc']
        sulc_data = np.concatenate(
            [load_surf_data(str(f)) for f in sulc_files])
        sulc_data_thresh = np.where(sulc_data > 0.0, 9.0, 7.0)

        # Define surface plot
        plot = Plot(
            lh_file, rh_file, layout='grid', size=(800, 550), zoom=1.8)
        plot.add_layer(
            sulc_data_thresh, cmap='Greys', color_range=(threshold, vmax),
            cbar=False)
        plot.add_layer(
            z_map_thresh_pos, cmap='Reds_r', color_range=(threshold, vmax),
            cbar_label='Audios > images')
        plot.add_layer(
            z_map_thresh_neg, cmap='Blues_r', color_range=(threshold, vmax),
            cbar_label='Images > audios')

        # Display the plot
        vdisplay = Xvfb()
        vdisplay.start()
        fig = plot.build(colorbar=False)
        fig.tight_layout()
        fig_file = output_sub_dir / f'{contrast_name}.png'
        plt.savefig(fig_file, dpi=300)
        fig_dict[sub][ses] = fig_file
        vdisplay.stop()

# Create empty figure
n_subjects = len(fig_dict)
n_sessions = max([len(elem) for elem in fig_dict.values()])
figsize = (n_subjects * 10, n_sessions)
fig, axs = plt.subplots(n_subjects, n_sessions, figsize=figsize)

# Add images
for ses_ix, (ses, fig_file) in enumerate(fig_dict['sub-SA27'].items()):
    img = mpimg.imread(fig_file)
    _ = axs[ses_ix].imshow(img)
    _ = axs[ses_ix].axis('off')
    title = sub + ', ' + ses
    _ = axs[ses_ix].set_title(title)

# # Add images
# for sub_ix, (sub, session_dict) in enumerate(fig_dict.items()):
#     for ses_ix, (ses, fig_file) in enumerate(session_dict.items()):
#         img = mpimg.imread(fig_file)
#         _ = axs[ses_ix][sub_ix].imshow(img)
#         _ = axs[ses_ix][sub_ix].axis('off')
#         title = sub + ', ' + ses
#         _ = axs[ses_ix][sub_ix].set_title(title, y=0.8)

# Add colorbars
fig.subplots_adjust(bottom=0.001)
cmap_1 = plt.cm.ScalarMappable(
    cmap='Reds_r', norm=plt.Normalize(vmin=threshold, vmax=vmax))
cmap_2 = plt.cm.ScalarMappable(
    cmap='Blues_r', norm=plt.Normalize(vmin=threshold, vmax=vmax))
cbar_ax_1 = fig.add_axes([0.4, 0.08, 0.2, 0.03])
cbar_ax_2 = fig.add_axes([0.4, 0.0, 0.2, 0.03])
cbar_1 = fig.colorbar(cmap_1, cax=cbar_ax_1, orientation='horizontal')
cbar_1.set_ticks([])
cbar_2 = fig.colorbar(cmap_2, cax=cbar_ax_2, orientation='horizontal')
cbar_1.set_label('Audio > image ($\it{Z}$)', x=-0.35, labelpad=-12)
cbar_2.set_label('Image > audio ($\it{Z}$)', x=-0.35, labelpad=-25)

# Adjust spacing and save
_ = plt.subplots_adjust(wspace=0.05, hspace=-0.6)
fig_file = output_dir / f'group_{contrast_name}_surf.png'
_ = plt.savefig(fig_file, dpi=300, bbox_inches='tight',
                facecolor='white', transparent=False)
