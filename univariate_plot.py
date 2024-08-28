from pathlib import Path
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain, plot_stat_map
from univariate import CONTRASTS, DERIVATIVES_DIR, SPACE, TASK, UNIVARIATE_DIR

TEMPLATEFLOW_DIR = DERIVATIVES_DIR / 'templateflow'

mpl.rcParams.update({"font.family": ["Open Sans"], "font.size": 12})

template_dir = TEMPLATEFLOW_DIR / f'tpl-{SPACE}'
template_filename = f'tpl-{SPACE}_res-01_T1w.nii.gz'
anat_file = template_dir / template_filename

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

            plt.subplots_adjust(wspace=0.05)

            plot = plot_glass_brain(clusters_file, display_mode='lyrz',
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
            plot.title(title, color='black', bgcolor='white', y=1.1)

    desc = auditory_contrast.replace('audios-', '')
    fig_filename = f'task-{TASK}_space-{SPACE}_desc-{desc}_clusters.png'
    fig_file = fig_dir / fig_filename
    fig.savefig(fig_file, dpi=600, bbox_inches='tight')
