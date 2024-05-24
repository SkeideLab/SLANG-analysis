from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.image import binarize_img, load_img, math_img
from nilearn.plotting import plot_stat_map

bids_dir = Path('/ptmp/aenge/slang/data')
derivatives_dir = bids_dir / 'derivatives'
results_dir = derivatives_dir / 'univariate'
output_dir = results_dir / 'figures'

task = 'language'
space = 'MNI152NLin2009cAsym'
contrasts = [
    'audios-noise',
    'audios-pseudo',
    'audios-pseudo-minus-noise',
    'audios-words',
    'audios-words-minus-noise',
    'audios-words-minus-pseudo',
    'images-noise',
    'images-pseudo',
    'images-pseudo-minus-noise',
    'images-words',
    'images-words-minus-noise',
    'images-words-minus-pseudo',
    'images-minus-audios'
]

anat_file = derivatives_dir / 'templateflow' / f'tpl-{space}' / \
    f'tpl-{space}_res-01_T1w.nii.gz'
anat_img = load_img(anat_file)

output_dir.mkdir(exist_ok=True)

# Contrast maps

for contrast in contrasts:

    for effect in ['0', '1', '2']:  # Intercept, linear, quadratic

        b_file = results_dir / \
            f'task-{task}_space-{space}_desc-{contrast}_b{effect}.nii.gz'
        b_img = load_img(b_file)

        cluster_file = results_dir / \
            f'task-{task}_space-{space}_desc-{contrast}_z{effect}-clusters.nii.gz'
        pos_file = results_dir / \
            f'task-{task}_space-{space}_desc-{contrast}_z{effect}-clusters-pos.nii.gz'
        neg_file = results_dir / \
            f'task-{task}_space-{space}_desc-{contrast}_z{effect}-clusters-neg.nii.gz'

        if cluster_file.exists():
            cluster_img = binarize_img(cluster_file)
            stat_img = math_img('img1 * img2', img1=b_img, img2=cluster_img)
        elif pos_file.exists() and neg_file.exists():
            pos_img = binarize_img(pos_file)
            neg_file = results_dir / \
                f'task-{task}_space-{space}_desc-{contrast}_z{effect}-clusters-neg.nii.gz'
            neg_img = binarize_img(neg_file)
            stat_img = math_img('(img1 + img2) * img3', img1=pos_img,
                                img2=neg_img, img3=b_img)
        else:
            stat_img = math_img('img - img', img=b_img)

        contrast_label = contrast.capitalize()

        if effect == '0':
            effect_label = 'intercept'
        elif effect == '1':
            effect_label = 'linear change'
        elif effect == '2':
            effect_label = 'quadratic change'

        title = f'{contrast_label}, {effect_label} (a.u.)'
        _ = plot_stat_map(stat_img, anat_img, cut_coords=4,
                          display_mode='z', title=None)

        figure_file = output_dir / \
            f'task-{task}_space-{space}_desc-{contrast}_z{effect}-clusters.png'
        plt.savefig(figure_file, dpi=1000)

# Line plot

contrast = 'audios-words'


def add_quadratic_line(intercept, linear, quadratic):
    """Plot a quadratic line. Adapted from https://stackoverflow.com/a/43811762"""
    axes = plt.gca()
    x_vals = np.arange(*axes.get_xlim(), 0.1)
    y_vals = intercept + linear * x_vals + quadratic * x_vals**2
    plt.plot(x_vals, y_vals, '--', color='white')


# Get location of significant linear cluster
stat_filename = f'task-{task}_space-{space}_desc-{contrast}_z1-clusters.nii.gz'
stat_img = load_img(results_dir / stat_filename)
data = stat_img.get_fdata()
cluster_ixs = np.where(data != 0)

# Extract single-subject betas
dfs = []
subject_dirs = sorted(list(results_dir.glob('sub-*')))
for subject_dir in subject_dirs:

    subject = subject_dir.name.replace('sub-', '')
    session_dirs = sorted(list(subject_dir.glob('ses-*')))

    for session_dir in session_dirs:

        session = session_dir.name.replace('ses-', '')
        raw_dir = bids_dir / subject_dir.name / session_dir.name
        scans_filename = f'sub-{subject}_ses-{session}_scans.tsv'
        scans_file = raw_dir / scans_filename
        scans_df = pd.read_csv(scans_file, sep='\t')
        scans_df = scans_df.query('filename.str.startswith("func")')
        assert len(scans_df) == 1
        acq_time = scans_df['acq_time'].values[0]

        func_dir = session_dir / 'func'
        beta_filename = f'sub-{subject}_ses-{session}_task-{task}_space-{space}_desc-{contrast}_effect_size.nii.gz'
        beta_file = func_dir / beta_filename
        betas = load_img(beta_file).get_fdata()[cluster_ixs]
        beta = betas.mean()
        df = pd.DataFrame({'subject': [subject],
                           'session': [session],
                           'acq_time': [acq_time],
                           'beta': [beta]})
        dfs.append(df)


df = pd.concat(dfs)
df['acq_time'] = pd.to_datetime(df['acq_time'])
df['time_diff'] = df['acq_time'] - df['acq_time'].min()
df['time'] = df['time_diff'].dt.days / 30.437  # Convert to months
df = df.reset_index()

b0_filename = f'task-{task}_space-{space}_desc-{contrast}_b0.nii.gz'
b0_img = load_img(results_dir / b0_filename)
b0_data = b0_img.get_fdata()[cluster_ixs]
b0 = b0_data.mean()

b1_filename = f'task-{task}_space-{space}_desc-{contrast}_b1.nii.gz'
b1_img = load_img(results_dir / b1_filename)
b1_data = b1_img.get_fdata()[cluster_ixs]
b1 = b1_data.mean()

b2_filename = f'task-{task}_space-{space}_desc-{contrast}_b2.nii.gz'
b2_img = load_img(results_dir / b2_filename)
b2_data = b2_img.get_fdata()[cluster_ixs]
b2 = b2_data.mean()

plt.style.use("dark_background")
plt.figure(figsize=(6, 3))
ax = sns.lineplot(df, x='time', y='beta', hue='subject', marker='o',
                  markeredgewidth=0, legend=False)
add_quadratic_line(b0, b1, b2)
plt.xlabel('Time (months)')
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
plt.ylabel('Beta (a.u.)')
plt.tight_layout()

plot_filename = f'task-{task}_space-{space}_desc-{contrast}_lines.png'
plt.savefig(output_dir / plot_filename, dpi=1000)
