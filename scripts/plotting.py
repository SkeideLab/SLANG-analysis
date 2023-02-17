from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from nilearn.surface import load_surf_data, vol_to_surf
from surfplot import Plot
from xvfbwrapper import Xvfb

bids_dir = Path('/ptmp/aenge/slang/data')
deriv_dir = bids_dir / 'derivatives'
fmriprep_dir = deriv_dir / 'fmriprep'
freesurfer_dir = fmriprep_dir / 'sourcedata/freesurfer'
lisa_dir = deriv_dir / 'lisa'

task = 'language'
run = '1'
space = 'T1w'
contrast = 'images-minus-audios'
hemi = 'lh'
surf = 'inflated'
vol_to_surf_interpolation = 'linear'
vol_to_surf_kind = 'depth'
p_thresh = 0.05

ru = 'run-' + run
tas = 'task-' + task
spa = 'space-' + space
des = 'desc-' + contrast

vdisplay = Xvfb()
vdisplay.start()

fig_dict = {}

subject_dirs = sorted(list(lisa_dir.glob('sub-*')))
for subject_dir in subject_dirs:
    sub = subject_dir.name

    surf_dir = freesurfer_dir / sub / 'surf'
    pial_file = surf_dir / f'{hemi}.pial'
    white_file = surf_dir / f'{hemi}.white'

    inflated_file = fmriprep_dir / sub / \
        f'anat/{sub}_{ru}_hemi-L_inflated.surf.gii'
    sulc_file = surf_dir / f'{hemi}.sulc'
    sulc_map = load_surf_data(str(sulc_file))
    sulc_map_thresh = np.where(sulc_map > 3.0, 0.8, 0.4)

    session_dirs = sorted(list(subject_dir.glob('ses-*')))
    if len(session_dirs) >= 2:
        fig_dict[sub] = {}
        for session_dir in session_dirs:
            ses = session_dir.name

            maps_dir = lisa_dir / sub / ses / 'func'

            fdr_file = maps_dir / f'{sub}_{ses}_{tas}_{ru}_{spa}_{des}_fdr.nii'
            fdr_map = vol_to_surf(
                fdr_file, pial_file, interpolation=vol_to_surf_interpolation,
                kind=vol_to_surf_kind, inner_mesh=white_file)

            fdr_thresh = 1 - p_thresh
            fdr_map_thresh = np.where(fdr_map > fdr_thresh, fdr_map, 0.0)

            # beta_file = maps_dir / f'{sub}_{ses}_{tas}_{ru}_{spa}_{des}_betas.nii'
            # beta_map = vol_to_surf(
            #     beta_file, pial_file, interpolation=vol_to_surf_interpolation,
            #     kind=vol_to_surf_kind, inner_mesh=white_file)

            # beta_map_thresh = np.where(fdr_map > fdr_thresh, beta_map, 0.0)

            plot = Plot(
                inflated_file, views='lateral', size=(800, 550), zoom=1.6)
            plot.add_layer(
                sulc_map_thresh, cmap='Greys', color_range=(0.0, 1.0),
                cbar=False)
            plot.add_layer(
                fdr_map_thresh, cmap='plasma', color_range=(fdr_thresh, 1.0))

            fig = plot.build(colorbar=False)
            fig.tight_layout()
            plot_dir = deriv_dir / 'surfplot' / sub / ses / 'func'
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig_file = plot_dir / \
                f'{sub}_{ses}_{tas}_{ru}_{spa}_{des}_plot.png'
            plt.savefig(fig_file, dpi=300)
            plt.close()

            fig_dict[sub][ses] = fig_file

            # from surfer import Brain
            # brain = Brain(
            #     sub, hemi, surf, cortex='classic', size=(800, 600),
            #     background='white', subjects_dir=freesurfer_dir)

            # if np.any(fdr_map > fdr_thresh):
            #     brain.add_data(
            #         fdr_map, min=0.95, max=1.0, thresh=0.95, colormap='YlOrRd')

            # plot_dir = deriv_dir / 'pysurfer' / sub / ses / 'func'
            # plot_dir.mkdir(parents=True, exist_ok=True)

            # plot_filename = f'{sub}_{ses}_{tas}_{ru}_{spa}_{des}_plot.png'
            # plot_file = plot_dir / plot_filename
            # brain.save_image(plot_file)

vdisplay.stop()

fig_dict_sa = {k: v for k, v in fig_dict.items() if k.startswith('sub-SA')}
fig_dict_so = {k: v for k, v in fig_dict.items() if k.startswith('sub-SO')}

groups_dict = {'SA': fig_dict_sa, 'SO': fig_dict_so}
for group, group_dict in groups_dict.items():

    n_subjects = len(group_dict)
    n_sessions = max([len(elem) for elem in group_dict.values()])
    figsize = (n_sessions * 2, n_subjects * 2)
    fig, axs = plt.subplots(n_subjects, n_sessions, figsize=figsize)
    _ = [ax.set_axis_off() for ax in axs.ravel()]

    for sub_ix, (sub, session_dict) in enumerate(group_dict.items()):
        for ses_ix, (ses, fig_file) in enumerate(session_dict.items()):
            img = mpimg.imread(fig_file)
            _ = axs[sub_ix][ses_ix].imshow(img)
            _ = axs[sub_ix][ses_ix].axis('off')
            title = sub + ', ' + ses
            _ = axs[sub_ix][ses_ix].set_title(title, y=0.9)

    _ = plt.subplots_adjust(wspace=0.05, hspace=-0.6)
    fig_file = deriv_dir / \
        f'surfplot/group-{group}_{tas}_{ru}_{spa}_{des}_plot.png'
    _ = plt.savefig(fig_file, dpi=300, bbox_inches='tight',
                    facecolor='white', transparent=False)

# vdisplay = Xvfb()
# vdisplay.start()

# from surfer import Brain
# from mayavi import mlab
# from tvtk.api import tvtk
# from tvtk.common import configure_input_data

# fig = mlab.figure()

# brain = Brain(
#     sub, hemi, surf, title, cortex='classic', background='white',
#     figure=fig, subjects_dir=freesurfer_dir)

# x, y, z = brain.geo[hemi].coords.T
# tris = brain.geo[hemi].faces

# hue = beta_map_thresh
# cmap = plt.cm.viridis
# colors = cmap(hue)[:, :3]

# alpha = fdr_map # (fdr_map + 3) / (fdr_map + 3)

# rgba_vals = np.concatenate((colors, alpha[:, None]), axis=1)

# mesh = mlab.pipeline.triangular_mesh_source(
#     x, y, z, tris, figure=fig)
# mesh.data.point_data.scalars.number_of_components = 4  # r, g, b, a
# mesh.data.point_data.scalars = (rgba_vals * 255).astype('ubyte')

# mapper = tvtk.PolyDataMapper()
# configure_input_data(mapper, mesh.data)
# actor = tvtk.Actor()
# actor.mapper = mapper
# fig.scene.add_actor(actor)

# mlab.savefig('test.png')
