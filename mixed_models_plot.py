from surfplot import Plot
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import load_surf_data
import numpy as np

surface = 'infl'  # or 'pial' or 'white' or 'thick
hemi = 'L'  # or 'R'
contrast = 'audios_words'
effect = 'intercept'
p_thresh = 0.001


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


fsaverage_dict = fetch_surf_fsaverage(mesh='fsaverage5')
hemi_nilearn = 'left' if hemi == 'L' else 'right'
anat_file = fsaverage_dict[f'{surface}_{hemi_nilearn}']
hemi_freesurfer = 'lh' if hemi == 'L' else 'rh'
surf = {f'surf_{hemi_freesurfer}': anat_file}

plot = Plot(**surf, size=(500, 200))

# curv_file = fsaverage_dict[f'sulc_{hemi_nilearn}']
# curv_map = load_surf_data(curv_file)
# curv_map_binary = np.where(curv_map > 0.0, 0.25, 0.4)
# _ = plot.add_layer(curv_map_binary, cmap='Greys_r', color_range=(0.0, 1.0),
#                    cbar=False)

stat_file = f"/Users/alexander/Research/slang/data/derivatives/julia/task-language_hemi-L_space-fsaverage5_desc-{contrast}_b_{effect}.gii"
stat_map = load_surf_data(stat_file)
vmax = my_ceil(np.max(np.abs(stat_map)), precision=2)
label = f'{contrast}\n{effect} ($z$)'
_ = plot.add_layer(stat_map, cmap='viridis', color_range=(-vmax, vmax),
                   zero_transparent=True, cbar_label=label)

p_file = f"/Users/alexander/Research/slang/data/derivatives/julia/task-language_hemi-L_space-fsaverage5_desc-{contrast}_p_{effect}.gii"
p_map = load_surf_data(p_file)
p_map[np.isnan(p_map)] = 1.0
p_map = np.where(p_map < p_thresh, 1.0, 0.0)
_ = plot.add_layer(p_map, cmap='Greys', alpha=1.0, color_range=(0.0, 1.25),
                   as_outline=True, cbar=False)
_ = plot.build()
