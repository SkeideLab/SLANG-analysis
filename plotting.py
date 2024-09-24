from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import load_img, math_img
from nilearn.plotting import plot_glass_brain
from similarity import ATLAS_FILE, SIMILARITY_DIR, get_roi_imgs
from stability import STABILITY_DIR
from univariate import CONTRASTS, SPACE, TASK, UNIVARIATE_DIR

mpl.rcParams.update({"font.family": ["Open Sans"], "font.size": 12})

STATS = {'z0': 'intercept', 'z1': 'linear change', 'z2': 'quadratic change'}

AUD_CONTRASTS = sorted([c for c in CONTRASTS if c.startswith('audios')])
VIS_CONTRASTS = sorted([c for c in CONTRASTS if c.startswith('images')])


def main():
    """Main function to plot all univariate and multivariate results."""

    meta_file = UNIVARIATE_DIR / f'task-{TASK}_space-{SPACE}_desc-metadata.tsv'
    meta_df = pd.read_csv(meta_file, sep='\t',
                          dtype={'subject': str, 'session': str})

    example_img = plot_univariate(meta_df)

    # example_img = load_img(UNIVARIATE_DIR / 'task-language_space-MNI152NLin2009cAsym_desc-audios-noise_b0.nii.gz')

    similarity_corr_df_filename = f'task-{TASK}_space-{SPACE}_desc-similarity_corrs.tsv'
    similarity_corr_df_file = SIMILARITY_DIR / similarity_corr_df_filename
    similarity_corr_df = pd.read_csv(similarity_corr_df_file, sep='\t')

    similarity_stat_df_filename = f'task-{TASK}_space-{SPACE}_desc-similarity_stats.tsv'
    similarity_stat_df_file = SIMILARITY_DIR / similarity_stat_df_filename
    similarity_stat_df = pd.read_csv(similarity_stat_df_file, sep='\t')

    plot_corrs(similarity_corr_df, similarity_stat_df, example_img,
               title_prefix='Audio-visual pattern similarity',
               output_dir=SIMILARITY_DIR, suffix='similarity')

    stability_corr_df_filename = f'task-{TASK}_space-{SPACE}_desc-stability_corrs.tsv'
    stability_corr_df_file = STABILITY_DIR / stability_corr_df_filename
    stability_corr_df = pd.read_csv(stability_corr_df_file, sep='\t')

    stability_stat_df_filename = f'task-{TASK}_space-{SPACE}_desc-stability_stats.tsv'
    stability_stat_df_file = STABILITY_DIR / stability_stat_df_filename
    stability_stat_df = pd.read_csv(stability_stat_df_file, sep='\t')

    plot_corrs(stability_corr_df, stability_stat_df, example_img,
               title_prefix='Pattern stability', output_dir=STABILITY_DIR,
               suffix='stability')


def plot_univariate(meta_df):
    """Plots univariate results (significant auditory and visual clusters for
    intercept, linear, and quadratic change) as glass brain plots."""

    fig_dir = UNIVARIATE_DIR / 'figures'
    fig_dir.mkdir(exist_ok=True)

    for aud_contrast, vis_contrast in zip(AUD_CONTRASTS, VIS_CONTRASTS):

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(20.0, 9.0))

        letter_ix = 0
        for row_ix, (stat, stat_label) in enumerate(STATS.items()):

            for col_ix, contrast in enumerate([aud_contrast, vis_contrast]):

                suffix = f'{stat}-clusters'
                clusters_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_{suffix}.nii.gz'
                clusters_file = UNIVARIATE_DIR / clusters_filename
                clusters_img = load_img(clusters_file)

                plt.subplots_adjust(wspace=0.05)

                plot = plot_glass_brain(clusters_img, display_mode='lyrz',
                                        colorbar=False, figure=fig,
                                        axes=axs[row_ix, col_ix],
                                        black_bg=False, cmap='blue_orange',
                                        vmin=-10.0, vmax=10.0, plot_abs=False)

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
                plot.title(title, color='black', bgcolor='white',
                           x=0.002, y=1.1)

        desc = aud_contrast.replace('audios-', '')
        fig_filename = f'task-{TASK}_space-{SPACE}_desc-{desc}_clusters.png'
        fig_file = fig_dir / fig_filename
        fig.savefig(fig_file, dpi=600, bbox_inches='tight')

        letter_ix = 6
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(19.163, 4.0))
        for col_ix, contrast in enumerate([aud_contrast, vis_contrast]):

            plt.sca(axs[col_ix])

            cluster_mask = math_img('img - img', img=clusters_img)
            max_cluster_size = 0
            max_cluster_stat = None

            for stat in STATS:

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
                    func_dir = UNIVARIATE_DIR / \
                        f'sub-{subject}/ses-{session}/func'
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
                for stat in STATS:
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

            desc = aud_contrast.replace('audios-', '')
            fig_filename = f'task-{TASK}_space-{SPACE}_desc-{desc}_lineplot.png'
            fig_file = fig_dir / fig_filename
            fig.savefig(fig_file, dpi=600, bbox_inches='tight')

    return clusters_img


def add_quadratic_line(intercept, linear, quadratic):
    """Plot a quadratic line. Adapted from https://stackoverflow.com/a/43811762"""

    axes = plt.gca()
    x_vals = np.arange(*axes.get_xlim(), 0.1)
    y_vals = intercept + linear * x_vals + quadratic * x_vals**2
    plt.plot(x_vals, y_vals, '--', color='black')


def plot_corrs(corr_df, stat_df, example_img, title_prefix, output_dir, suffix):

    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    roi_imgs = get_roi_imgs(ATLAS_FILE, ref_img=example_img)

    for contrast_label, contrast_df in corr_df.groupby('contrast_label',
                                                       observed=False):

        roi_labels = contrast_df['roi_label'].unique()
        contrast_df['roi_label'] = contrast_df['roi_label'].\
            astype('category').\
            cat.set_categories(roi_labels)

        n_rois = len(roi_labels)
        n_cols = 2
        n_rows = n_rois // 2
        figsize = (n_cols * 8.0, n_rois * 2.0)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

        axs = np.ravel(axs)
        letter_ix = 0
        for ix, (ax, (roi_label, roi_df)) in \
                enumerate(zip(axs, contrast_df.groupby('roi_label', observed=False))):

            roi_img = roi_imgs[roi_label]
            roi_ax_left = 0.4 if ix % 2 == 0 else 0.9
            roi_ax_bottom = 0.85 - (ix // 2) / n_rows
            roi_ax_width = 0.09
            roi_ax_height = 0.09
            roi_ax = fig.add_axes([roi_ax_left, roi_ax_bottom,
                                   roi_ax_width, roi_ax_height])
            plot_glass_brain(roi_img, display_mode='z', colorbar=False,
                             black_bg=False, axes=roi_ax)
            roi_title = roi_label.\
                replace('audios-', 'spoken\n').\
                replace('images-', 'written\n').\
                replace('noise', 'low-level').\
                replace('pseudo', 'pseudow.').\
                replace('words', 'word')
            if '-left' in roi_title:
                roi_title = roi_title.replace('-left', '')
                roi_title = f'Left {roi_title}'
            elif '-right' in roi_title:
                roi_title = roi_title.replace('-right', '')
                roi_title = f'Right {roi_title}'
            roi_title = f'{roi_title} ROI'
            roi_ax.set_title(roi_title)

            plt.sca(ax)
            sns.lineplot(roi_df, x='time', y='r', hue='subject', palette='hls',
                         marker='o', markeredgewidth=0, legend=False)

            this_stat_df = stat_df.query(f'contrast_label == "{contrast_label}" & ' +
                                         f'roi_label == "{roi_label}"')
            intercept = this_stat_df.loc[this_stat_df['effect']
                                         == 'intercept', 'beta'].values[0]
            linear = this_stat_df.loc[this_stat_df['effect']
                                      == 'linear', 'beta'].values[0]
            quadratic = this_stat_df.loc[this_stat_df['effect']
                                         == 'quadratic', 'beta'].values[0]
            add_quadratic_line(intercept, linear, quadratic)

            letter = ascii_uppercase[letter_ix]
            letter_ix += 1
            title = f'$\\bf{{{letter}}}$'
            plt.title(title, color='black', x=-0.095, y=0.93)

            plt.xlim(-1.0, 17.0)
            plt.ylim(-1.1, 1.1)
            plt.xlabel('Time (months)')
            plt.ylabel('Correlation coefficient')

            if ix == 0:
                pair_title_label = contrast_label.\
                    replace('images-', 'written ').\
                    replace('audios-', 'spoken ').\
                    replace('noise', 'low-level').\
                    replace('pseudo', 'pseudowords').\
                    replace('-minus-', ' vs. ')
                pair_title = f'{title_prefix} for {pair_title_label}'
                plt.annotate(pair_title, xy=(0.01, 0.93),
                             xycoords='axes fraction', size=14.0)

        plt.tight_layout()

        fig_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast_label}_{suffix}.png'
        fig_file = fig_dir / fig_filename
        fig.savefig(fig_file, dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    main()
