import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bids import BIDSLayout
from juliacall import Main as jl
from nilearn.image import math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.reporting import get_clusters_table
from scipy.stats import norm
from univariate import (BIDS_DIR, CONTRASTS, DERIVATIVES_DIR, DF_QUERY,
                        FD_THRESHOLD, FMRIPREP_DIR, HRF_MODEL, N_JOBS,
                        PYBIDS_DIR, SPACE, TASK, UNIVARIATE_DIR,
                        compute_beta_img, load_df, run_glms)

SIMILARITY_DIR = DERIVATIVES_DIR / 'similarity'
TEMPLATEFLOW_DIR = DERIVATIVES_DIR / 'templateflow'

# MNI Version of the Glasser et al. (2016) atlas, downloaded from
# https://figshare.com/articles/dataset/2016_Glasser_MMP1_0_Cortical_Atlases/24431146
TEMPLATE_DIR = TEMPLATEFLOW_DIR / f'tpl-{SPACE}'
ATLAS_FILE = TEMPLATE_DIR / f'glasser_{SPACE}_labels_p20.nii.gz'

# Inpupt parameters: First-level GLM + similarity analysis
SMOOTHING_FWHM = None
CONTRAST_PAIRS = {
    'noise': (('images_noise', None),
              ('audios_noise', None)),
    'pseudo': (('images_pseudo', None),
               ('audios_pseudo', None)),
    'words': (('images_words', None),
              ('audios_words', None)),
    'words-minus-pseudo': (('images_words', 'images_pseudo'),
                           ('audios_words', 'audios_pseudo')),
    'pseudo-minus-noise': (('images_pseudo', 'images_noise'),
                           ('audios_pseudo', 'audios_noise'))}

# Inpupt parameters: Anatomical regions of interest
GLASSER_ROIS = {'pSTS': (128, 129, 130, 176),  # STSda, STSdp, STSvp, STSva
                'vOT': (7, 18, 163)}  # 8th Visual Area, FFC, Ventral Visual Complex

# Inpupt parameters: Linear mixed models
FORMULA = 'r ~ time + time2 + (time + time2 | subject)'


def main():
    """Main function for running the full similarity analysis."""

    # Load BIDS structure
    layout = BIDSLayout(BIDS_DIR, derivatives=FMRIPREP_DIR,
                        database_path=PYBIDS_DIR)

    # Fit first-level GLM, separately for each subject and session
    glms, mask_imgs, percs_non_steady, percs_outliers, residuals_files = \
        run_glms(BIDS_DIR, FMRIPREP_DIR, PYBIDS_DIR, TASK, SPACE, FD_THRESHOLD,
                 HRF_MODEL, SMOOTHING_FWHM, SIMILARITY_DIR, N_JOBS)

    df = load_df(layout, TASK, percs_non_steady, percs_outliers, DF_QUERY)
    subjects = df['subject'].tolist()
    sessions = df['session'].tolist()
    good_ixs = list(df.index)

    glms = [glms[ix] for ix in good_ixs]
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]
    residuals_files = [residuals_files[ix] for ix in good_ixs]

    atlas_img = resample_to_img(ATLAS_FILE, mask_imgs[0],
                                interpolation='nearest')
    anat_roi_maskers = get_anat_rois(atlas_img)
    func_roi_maskers = get_func_rois()
    roi_maskers = {**anat_roi_maskers, **func_roi_maskers}

    corr_df = correlate_contrast_pairs(subjects, sessions, glms, roi_maskers)

    df = pd.merge(df, corr_df, on=['subject', 'session'])
    df_filename = f'task-{TASK}_space-{SPACE}_desc-correlations.tsv'
    df_file = SIMILARITY_DIR / df_filename
    df.to_csv(df_file, sep='\t', index=False, float_format='%.4f')

    # # Read previously saved data frame
    # df = pd.read_csv(df_file, sep='\t')

    res_df = run_similarity_stats(df, roi_maskers)
    res_df_filename = f'task-{TASK}_space-{SPACE}_desc-stats.tsv'
    res_df_file = SIMILARITY_DIR / res_df_filename
    res_df.to_csv(res_df_file, sep='\t', index=False, float_format='%.4f')

    # for conditions in condition_pairs:

    #     for glasser_roi_label in glasser_roi_labels:

    #         plot_df = df.query(f'condition_a == "{conditions[0]}" & ' +
    #                            f'condition_b == "{conditions[1]}" & ' +
    #                            f'roi_label == "{glasser_roi_label}"')

    #         sns.lineplot(data=plot_df, x='time', y='r', hue='subject',
    #                      marker='o', legend=False)
    #         plt.ylim(-1.0, 1.0)
    #         plt.title(f'Correlation between "{conditions[0]}" and ' +
    #                   f'"{conditions[1]}" in {glasser_roi_label}')
    #         plt.xlabel('Time (months)')
    #         plt.ylabel('Correlation')
    #         plt.show()


def get_anat_rois(atlas_img):
    """Returns a dictionary of NiftiMaskers for anatomical ROIs in an atlas."""

    roi_maskers = {}
    for roi_label, roi_nos in GLASSER_ROIS.items():
        roi_img = math_img(f'np.sum(img == roi for roi in {roi_nos})',
                           img=atlas_img)
        roi_masker = NiftiMasker(mask_img=roi_img, standardize=True)
        roi_maskers[roi_label] = roi_masker

    return roi_maskers


def get_func_rois():
    """Returns a dictionary of NiftiMaskers for functional ROIs from the
    univariate analysis.

    For each contrast, the top 2 largest clusters are extracted (typically the
    left and right auditory cortex for auditory contrasts and the left and
    right visual cortex for visual contrasts)."""

    roi_maskers = {}
    for contrast in CONTRASTS:

        if 'minus' in contrast:
            continue

        clusters_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_z0-clusters.nii.gz'
        clusters_file = UNIVARIATE_DIR / clusters_filename

        for ix in range(1, 3):  # Use top 2 largest clusters only
            cluster_img = math_img(f'img == {ix}', img=clusters_file)
            cluster_hemi = get_cluster_hemi(cluster_img)
            roi_label = f'{contrast}-{cluster_hemi}'
            roi_masker = NiftiMasker(mask_img=cluster_img, standardize=True)
            roi_maskers[roi_label] = roi_masker

    return roi_maskers


def get_cluster_hemi(one_cluster_img):
    """Returns the hemisphere (left or right) of the single (!) cluster in img."""

    y = get_clusters_table(one_cluster_img, stat_threshold=0.1)['X'][0]

    if y < 0:
        return 'left'
    else:
        return 'right'


def correlate_contrast_pairs(subjects, sessions, glms, roi_maskers):
    """Correlate beta values for pairs of contrasts in all ROIs."""

    corr_dfs = []
    for subject, session, glm in zip(subjects, sessions, glms):

        for pair_label, pair in CONTRAST_PAIRS.items():

            assert len(pair) == 2

            for roi_label, roi_masker in roi_maskers.items():

                condition_betas = []

                for conditions in pair:

                    beta_img = \
                        compute_beta_img(glm,
                                         conditions_plus=(conditions[0],),
                                         conditions_minus=(conditions[1],))

                    betas = roi_masker.fit_transform(beta_img)[0]
                    condition_betas.append(betas)

                corr = np.corrcoef(condition_betas)[0, 1]

                corr_df = pd.DataFrame({'subject': subject,
                                        'session': session,
                                        'pair_label': pair_label,
                                        'roi_label': roi_label,
                                        'r': corr},
                                       index=[0])
                corr_dfs.append(corr_df)

    return pd.concat(corr_dfs, axis=0, ignore_index=True)


def run_similarity_stats(df, roi_maskers):
    """Run linear mixed models on the correlation data, separately for each
    pair of condition and region of interest."""

    res_dfs = []
    for pair_label in CONTRAST_PAIRS.keys():
        for roi_label in roi_maskers.keys():
            model_df = df.query(f'pair_label == "{pair_label}" & ' +
                                f'roi_label == "{roi_label}"')
            bs, zs = fit_mixed_model(FORMULA, model_df)
            res_df = pd.DataFrame({'pair_label': pair_label,
                                   'roi_label': roi_label,
                                   'effect': ['intercept', 'linear', 'quadratic'],
                                   'beta': bs,
                                   'z': zs})
            res_dfs.append(res_df)

    res_df = pd.concat(res_dfs, axis=0, ignore_index=True)
    res_df['p'] = norm.sf(np.abs(res_df['z'])) * 2

    return res_df


def fit_mixed_model(formula, df):
    """Fits a mixed model to a DataFrame using the `MixedModels package in Julia."""

    model_cmd = f"""
        using MixedModels
        using Suppressor

        function fit_mixed_model(df)
          fml = @formula({formula})
          mod = @suppress fit(MixedModel, fml, df)
          bs = mod.beta
          zs = mod.beta ./ mod.stderror
        return bs, zs
        end"""
    fit_mixed_model_julia = jl.seval(model_cmd)

    return fit_mixed_model_julia(df)


if __name__ == '__main__':
    main()
