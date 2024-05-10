from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import fdr_threshold
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import plot_stat_map

contrasts = ['audios_noise',
             'audios_pseudo',
             'audios_words',
             'images_noise',
             'images_pseudo',
             'images_words',
             'images_pseudo_minus_noise',
             'images_words_minus_pseudo']
contrasts = ['images_minus_audios']
niftis_dir = Path('/ptmp/aenge/slang/data/derivatives/julia_vol/nifti')
figures_dir = Path('/ptmp/aenge/slang/data/derivatives/julia_vol/figures')

niftis_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

for contrast in contrasts:

    beta_file = f'/ptmp/aenge/slang/data/derivatives/nilearn_vol/sub-SA03/ses-01/func/sub-SA03_ses-01_task-language_space-MNI152NLin2009cAsym_desc-{contrast}_effect_size.nii.gz'
    beta_img = load_img(beta_file)

    anat_file = '/ptmp/aenge/slang/data/derivatives/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz'
    anat_img = resample_to_img(anat_file, beta_img)

    gm_file = '/ptmp/aenge/slang/data/derivatives/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz'
    gm_img = resample_to_img(gm_file, beta_img)

    df_file = f'/ptmp/aenge/slang/data/derivatives/julia_vol/sub-group_task-language_space-MNI152NLin2009cAsym_desc-{contrast}_lmm.tsv'

    df = pd.read_csv(df_file, sep='\t')

    # effects = ['m0_b0', 'm0_z0', 'm0_p0', 'm1_b0', 'm1_b1', 'm1_z0', 'm1_z1',
    #            'm1_p0', 'm1_p1', 'm1_m0_chisq', 'm1_m0_dof', 'm1_m0_pchisq',
    #            'm2_b0', 'm2_b1', 'm2_b2', 'm2_z0', 'm2_z1', 'm2_z2', 'm2_p0',
    #            'm2_p1', 'm2_p2', 'm2_m0_chisq', 'm2_m0_dof', 'm2_m0_pchisq',
    #            'm2_m1_chisq', 'm2_m1_dof', 'm2_m1_pchisq']
    effects = ['m2_z0', 'm2_z1', 'm2_z2']

    for effect in effects:
        threshold = fdr_threshold(df[effect], alpha=0.05)
        df[effect] = df[effect].where(df[effect].abs() > threshold, 0.0)

    stat_arrs = {effect: np.zeros(beta_img.shape).squeeze()
                 for effect in effects}

    for row in df.iterrows():

        ixs = row[1][['x', 'y', 'z']].values.astype(int) - 1

        for effect in effects:
            stat_arrs[effect][tuple(ixs)] = row[1][effect]

    for effect, stat_arr in stat_arrs.items():

        stat_img = nib.Nifti1Image(stat_arr, beta_img.affine)

        stat_filename = f'task-language_space-MNI152NLin2009cAsym_desc-{contrast}_{effect}.nii.gz'
        stat_file = niftis_dir / stat_filename
        stat_img.to_filename(stat_file)

        title = f'{contrast} {effect}'
        _ = plot_stat_map(stat_img, bg_img=anat_img,
                          cut_coords=np.arange(-18, 46, 8),
                          display_mode='z', title=title, black_bg=True)

        figure_filename = f'task-language_space-MNI152NLin2009cAsym_desc-{contrast}_{effect}.png'
        output_file = figures_dir / figure_filename
        # _ = plt.savefig(output_file, dpi=300)
