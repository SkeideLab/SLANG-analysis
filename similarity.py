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
                        compute_beta_img, load_meta_df, run_glms)

# Input parameters: Input/output directories
SIMILARITY_DIR = DERIVATIVES_DIR / 'similarity'
TEMPLATEFLOW_DIR = DERIVATIVES_DIR / 'templateflow'

# MNI Version of the Glasser et al. (2016) atlas, downloaded from
# https://figshare.com/articles/dataset/2016_Glasser_MMP1_0_Cortical_Atlases/24431146
TEMPLATE_DIR = TEMPLATEFLOW_DIR / f'tpl-{SPACE}'
ATLAS_FILE = TEMPLATE_DIR / f'glasser_{SPACE}_labels_p20.nii.gz'

# Inpupt parameters: First-level GLM
BLOCKWISE = False
SMOOTHING_FWHM = None
SAVE_RESIDUALS = False

# Input parameters: Similarity analysis
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

    layout = BIDSLayout(BIDS_DIR, derivatives=FMRIPREP_DIR,
                        database_path=PYBIDS_DIR)

    glms, mask_imgs, percs_non_steady, percs_outliers = \
        run_glms(BIDS_DIR, FMRIPREP_DIR, PYBIDS_DIR, TASK, SPACE, BLOCKWISE,
                 FD_THRESHOLD, HRF_MODEL, SMOOTHING_FWHM, SIMILARITY_DIR,
                 SAVE_RESIDUALS, N_JOBS)

    meta_df = load_meta_df(layout, TASK, percs_non_steady, percs_outliers,
                           DF_QUERY)
    meta_df = meta_df[['subject', 'session', 'n_sessions', 'perc_non_steady',
                       'perc_outliers', 'time', 'time2']]
    subjects = meta_df['subject'].tolist()
    sessions = meta_df['session'].tolist()
    good_ixs = list(meta_df.index)

    glms = [glms[ix] for ix in good_ixs]
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]

    roi_maskers = get_roi_maskers(ATLAS_FILE, ref_img=mask_imgs[0])
    anat_roi_labels = list(GLASSER_ROIS.keys())

    corr_df = compute_similarity(subjects, sessions, glms,
                                 roi_maskers, anat_roi_labels)

    SIMILARITY_DIR.mkdir(exist_ok=True, parents=True)
    corr_df = pd.merge(meta_df, corr_df, on=['subject', 'session'])
    corr_df_filename = f'task-{TASK}_space-{SPACE}_desc-similarity_corrs.tsv'
    corr_df_file = SIMILARITY_DIR / corr_df_filename
    corr_df.to_csv(corr_df_file, sep='\t', index=False, float_format='%.4f')

    # corr_df = pd.read_csv(corr_df_file, sep='\t', dtype={'session': str})

    stat_df = run_similarity_stats(corr_df, roi_maskers)
    stat_df_filename = f'task-{TASK}_space-{SPACE}_desc-similarity_stats.tsv'
    stat_df_file = SIMILARITY_DIR / stat_df_filename
    stat_df.to_csv(stat_df_file, sep='\t', index=False, float_format='%.4f')


def get_roi_maskers(atlas_file, ref_img):
    """Returns a dictionary of ROI maskers for anatomical and functional ROIs."""

    roi_imgs = get_roi_imgs(atlas_file, ref_img)
    return make_roi_maskers(roi_imgs)


def get_roi_imgs(atlas_file, ref_img):
    """Returns a dictionary of ROI mask images for anatomical and functional ROIs."""

    atlas_img = resample_to_img(atlas_file, ref_img, interpolation='nearest')
    anat_roi_imgs = get_anat_roi_imgs(atlas_img)
    func_roi_imgs = get_func_roi_imgs()

    return {**anat_roi_imgs, **func_roi_imgs}


def get_anat_roi_imgs(atlas_img):
    """Returns a dictionary of ROI mask images for anatomical ROIs in an atlas."""

    roi_imgs = {}
    for roi_label, roi_nos in GLASSER_ROIS.items():
        roi_img = math_img(f'np.sum(img == roi for roi in {roi_nos})',
                           img=atlas_img)
        roi_imgs[roi_label] = roi_img

    return roi_imgs


def get_func_roi_imgs():
    """Returns a dictionary of ROI mask images for functional ROIs from the
    univariate analysis.

    For each contrast, the top 2 largest clusters are extracted (typically the
    left and right auditory cortex for auditory contrasts and the left and
    right visual cortex for visual contrasts)."""

    roi_imgs = {}
    for contrast in CONTRASTS:

        if 'minus' in contrast:
            continue

        clusters_filename = f'task-{TASK}_space-{SPACE}_desc-{contrast}_z0-clusters.nii.gz'
        clusters_file = UNIVARIATE_DIR / clusters_filename

        for ix in range(1, 3):  # Use top 2 largest clusters only
            roi_img = math_img(f'img == {ix}', img=clusters_file)
            roi_hemi = get_roi_hemi(roi_img)
            roi_label = f'{contrast}-{roi_hemi}'
            roi_imgs[roi_label] = roi_img

    return roi_imgs


def get_roi_hemi(roi_img):
    """Returns the hemisphere (left or right) of the single (!) ROI cluster in img."""

    y = get_clusters_table(roi_img, stat_threshold=0.1)['X'][0]

    if y < 0:
        return 'left'
    else:
        return 'right'


def make_roi_maskers(roi_imgs):
    """Returns a dictionary of ROI maskers from binary ROI mask images."""

    roi_maskers = {}
    for roi_label, roi_img in roi_imgs.items():
        roi_masker = NiftiMasker(roi_img, standardize=True)
        roi_masker.fit()
        roi_maskers[roi_label] = roi_masker

    return roi_maskers


def compute_similarity(subjects, sessions, glms, roi_maskers, anat_roi_labels):
    """Correlate beta values for pairs of contrasts in all ROIs."""

    corr_dfs = []
    for subject, session, glm in zip(subjects, sessions, glms):

        for contrast_label, contrast_pair in CONTRAST_PAIRS.items():

            assert len(contrast_pair) == 2

            condition = contrast_label.split('-')[0]
            func_roi_labels = [roi_label for roi_label in roi_maskers.keys()
                               if f'-{condition}-' in roi_label]
            roi_labels = anat_roi_labels + func_roi_labels
            roi_maskers_ = {roi_label: roi_masker
                            for roi_label, roi_masker in roi_maskers.items()
                            if roi_label in roi_labels}

            for roi_label, roi_masker in roi_maskers_.items():

                condition_betas = []

                for conditions in contrast_pair:

                    beta_img = \
                        compute_beta_img(glm,
                                         conditions_plus=(conditions[0],),
                                         conditions_minus=(conditions[1],))

                    betas = roi_masker.fit_transform(beta_img)[0]
                    condition_betas.append(betas)

                corr = np.corrcoef(condition_betas)[0, 1]

                corr_df = pd.DataFrame({'subject': subject,
                                        'session': session,
                                        'contrast_label': contrast_label,
                                        'roi_label': roi_label,
                                        'r': corr},
                                       index=[0])
                corr_dfs.append(corr_df)

    return pd.concat(corr_dfs, ignore_index=True)


def run_similarity_stats(corr_df, roi_maskers):
    """Run linear mixed models on the correlation data, separately for each
    pair of condition and region of interest."""

    stat_dfs = []
    for contrast_label, contrast_corr_df in corr_df.groupby('contrast_label'):
        for roi_label, roi_corr_df in contrast_corr_df.groupby('roi_label'):
            bs, zs = fit_mixed_model(FORMULA, roi_corr_df)
            stat_df = pd.DataFrame({'contrast_label': contrast_label,
                                    'roi_label': roi_label,
                                    'effect': ['intercept', 'linear', 'quadratic'],
                                    'beta': bs,
                                    'z': zs})
            stat_dfs.append(stat_df)

    stat_df = pd.concat(stat_dfs, axis=0, ignore_index=True)
    stat_df['p'] = norm.sf(np.abs(stat_df['z'])) * 2

    return stat_df


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
