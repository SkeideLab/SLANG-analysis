from pathlib import Path
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bids import BIDSLayout
from nilearn.image import load_img, math_img
from nilearn.plotting import plot_glass_brain
from univariate import CONTRASTS, DERIVATIVES_DIR, SPACE, TASK, UNIVARIATE_DIR


def add_quadratic_line(intercept, linear, quadratic):
    """Plot a quadratic line. Adapted from https://stackoverflow.com/a/43811762"""

    axes = plt.gca()
    x_vals = np.arange(*axes.get_xlim(), 0.1)
    y_vals = intercept + linear * x_vals + quadratic * x_vals**2
    plt.plot(x_vals, y_vals, '--', color='black')


TEMPLATEFLOW_DIR = DERIVATIVES_DIR / 'templateflow'

mpl.rcParams.update({"font.family": ["Open Sans"], "font.size": 12})

template_dir = TEMPLATEFLOW_DIR / f'tpl-{SPACE}'
template_filename = f'tpl-{SPACE}_res-01_T1w.nii.gz'
anat_file = template_dir / template_filename

meta_file = UNIVARIATE_DIR / f'task-{TASK}_space-{SPACE}_desc-metadata.tsv'
meta_df = pd.read_csv(meta_file, sep='\t',
                      dtype={'subject': str, 'session': str})

stats = {'z0': 'intercept', 'z1': 'linear change', 'z2': 'quadratic change'}

fig_dir = UNIVARIATE_DIR / 'figures'
fig_dir.mkdir(exist_ok=True)

auditory_contrasts = sorted([c for c in CONTRASTS if c.startswith('audios')])
visual_contrasts = sorted([c for c in CONTRASTS if c.startswith('images')])

for auditory_contrast, visual_contrast in zip(auditory_contrasts, visual_contrasts):

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20.0, 9.0))

    letter_ix = 0
    for row_ix, (stat, stat_label) in enumerate(stats.items()):

        for col_ix, contrast in enumerate([auditory_contrast, visual_contrast]):

            suffix = f'{stat}-clusters'
            clusters_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_{suffix}.nii.gz'
            clusters_file = UNIVARIATE_DIR / clusters_filename
            clusters_img = load_img(clusters_file)

            plt.subplots_adjust(wspace=0.05)

            plot = plot_glass_brain(clusters_img, display_mode='lyrz',
                                    colorbar=False, figure=fig,
                                    axes=axs[row_ix, col_ix], black_bg=False,
                                    cmap='blue_orange', vmin=-10.0, vmax=10.0,
                                    plot_abs=False)

            contrast_label = contrast.\
                replace('-', ' ').\
                replace('minus', 'vs.').\
                replace('audios', 'Spoken').\
                replace('images', 'Written').\
                replace('noise', 'low-level').\
                replace('pseudo', 'pseudowords')
            letter = ascii_uppercase[letter_ix]
            letter_ix += 1
            title = f'$\\bf{{{letter}}}$  {contrast_label}, {stat_label}'
            plot.title(title, color='black', bgcolor='white', x=0.002, y=1.1)

    desc = auditory_contrast.replace('audios-', '')
    fig_filename = f'task-{TASK}_space-{SPACE}_desc-{desc}_clusters.png'
    fig_file = fig_dir / fig_filename
    fig.savefig(fig_file, dpi=600, bbox_inches='tight')

    # Line plot for the largest significant cluster (from linear or quadratic change)
    letter_ix = 6
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19.163, 4.0))
    for col_ix, contrast in enumerate([auditory_contrast, visual_contrast]):

        plt.sca(axs[col_ix])

        # Start with empty mask
        cluster_mask = math_img('img - img', img=clusters_img)
        max_cluster_size = 0
        max_cluster_stat = None

        for stat in stats:

            if stat == 'z0':
                continue

            suffix = f'{stat}-clusters'
            clusters_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_{suffix}.nii.gz'
            clusters_file = UNIVARIATE_DIR / clusters_filename
            clusters_img = load_img(clusters_file)
            clusters_data = clusters_img.get_fdata()
            clusters_data = clusters_data[clusters_data != 0.0]

            if not clusters_data.any():
                continue

            cluster_ixs, cluster_counts = np.unique(clusters_data,
                                                    return_counts=True)
            if cluster_counts.max() > max_cluster_size:
                max_cluster_ix = cluster_ixs[np.argmax(cluster_counts)]
                cluster_mask = math_img(f'img == {max_cluster_ix}',
                                        img=clusters_img)
                max_cluster_stat = stat

        if max_cluster_stat is None:

            plt.axis('off')

        else:

            fig_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_cluster-mask.png'
            fig_file = fig_dir / fig_filename
            plot_glass_brain(cluster_mask, output_file=fig_file,
                             display_mode='lyrz', colorbar=False, black_bg=False,
                             title=f'{contrast} cluster mask (from {max_cluster_stat})')

            voxel_ixs = np.where(cluster_mask.get_fdata())

            beta_dfs = []
            for row in meta_df.itertuples():

                subject = row.subject
                session = row.session
                func_dir = UNIVARIATE_DIR / f'sub-{subject}/ses-{session}/func'
                beta_filename = f'sub-{subject}_ses-{session}_task-{TASK}_space-{SPACE}_desc-{contrast}_effect_size.nii.gz'
                beta_file = func_dir / beta_filename
                beta_img = load_img(beta_file)
                betas = beta_img.get_fdata()[voxel_ixs]
                beta_df = pd.DataFrame({'subject': subject,
                                        'session': session,
                                        'time': row.time,
                                        'time2': row.time2,
                                        'beta': betas.mean()},
                                       index=[0])
                beta_dfs.append(beta_df)

            beta_df = pd.concat(beta_dfs, ignore_index=True)

            sns.lineplot(beta_df, x='time', y='beta', hue='subject',
                         palette='hls', marker='o', markeredgewidth=0,
                         legend=False)

            plt.subplots_adjust(wspace=0.15)
            plt.xlim(-1.0, 17.0)
            plt.xlabel('Time (months)')
            plt.ylabel('BOLD amplitude (a.u.)')

            bs = []
            for stat in stats:
                b_stat = stat.replace('z', 'b')
                b_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_{b_stat}.nii.gz'
                b_file = UNIVARIATE_DIR / b_filename
                b_img = load_img(b_file)
                b_vals = b_img.get_fdata()[voxel_ixs]
                bs.append(b_vals.mean())
            add_quadratic_line(*bs)

            letter = ascii_uppercase[letter_ix]
            letter_ix += 1
            title = f'$\\bf{{{letter}}}$'
            plt.title(title, color='black', x=-0.08, y=0.93)

        fig_filename = f'task-{TASK}_space-{SPACE}_desc-{desc}_lineplot.png'
        fig_file = fig_dir / fig_filename
        fig.savefig(fig_file, dpi=600, bbox_inches='tight')
