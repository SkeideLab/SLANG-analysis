from itertools import combinations
from warnings import warn

import numpy as np
import pandas as pd
from bids import BIDSLayout
from juliacall import Main as jl
from scipy.stats import norm
from similarity import ATLAS_FILE, FORMULA, GLASSER_ROIS, get_roi_maskers
from univariate import (BIDS_DIR, CONTRASTS, DERIVATIVES_DIR, DF_QUERY,
                        FD_THRESHOLD, FMRIPREP_DIR, PYBIDS_DIR, SPACE, TASK,
                        compute_beta_img, load_meta_df, run_glms)

# Input parameters: Input/output directories
STABILITY_DIR = DERIVATIVES_DIR / 'stability'

# Input parameters: First-level GLM
BLOCKWISE = True
HRF_MODEL = 'glover'
SMOOTHING_FWHM = None
SAVE_RESIDUALS = False
N_JOBS = 4


def main():
    """Main function for running the full pattern stability analysis."""

    # Load BIDS structure
    layout = BIDSLayout(BIDS_DIR, derivatives=FMRIPREP_DIR,
                        database_path=PYBIDS_DIR)

    # Fit first-level GLM, separately for each subject and session
    glms, mask_imgs, percs_non_steady, percs_outliers = \
        run_glms(BIDS_DIR, FMRIPREP_DIR, PYBIDS_DIR, TASK, SPACE, BLOCKWISE,
                 FD_THRESHOLD, HRF_MODEL, SMOOTHING_FWHM, STABILITY_DIR,
                 SAVE_RESIDUALS, N_JOBS)

    # Load metadata
    meta_df, good_ixs = load_meta_df(layout, TASK, percs_non_steady,
                                     percs_outliers, DF_QUERY)
    meta_df = meta_df.loc[good_ixs]
    meta_df = meta_df[['subject', 'session', 'n_sessions', 'perc_non_steady',
                       'perc_outliers', 'time', 'time2']]
    subjects = meta_df['subject'].tolist()
    sessions = meta_df['session'].tolist()

    # Exclude subjects/session that don't match the query
    glms = [glms[ix] for ix in good_ixs]
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]

    # Get anatomical and functional regions of interest
    roi_maskers = get_roi_maskers(ATLAS_FILE, ref_img=mask_imgs[0])
    anat_roi_labels = list(GLASSER_ROIS.keys())

    # Run pattern stability analysis
    corr_df = compute_stability(subjects, sessions, glms,
                                roi_maskers, anat_roi_labels)

    # Save results
    STABILITY_DIR.mkdir(exist_ok=True, parents=True)
    corr_df = pd.merge(meta_df, corr_df, on=['subject', 'session'])
    corr_df_filename = f'task-{TASK}_space-{SPACE}_desc-stability_corrs.tsv'
    corr_df_file = STABILITY_DIR / corr_df_filename
    corr_df.to_csv(corr_df_file, sep='\t', index=False, float_format='%.4f')

    # # Load previously saved results
    # corr_df = pd.read_csv(corr_df_file, sep='\t')

    # Run statistical analysis
    stat_df = run_stability_stats(corr_df)
    stat_df_filename = f'task-{TASK}_space-{SPACE}_desc-stability_stats.tsv'
    stat_df_file = STABILITY_DIR / stat_df_filename
    stat_df.to_csv(stat_df_file, sep='\t', index=False, float_format='%.4f')


def compute_stability(subjects, sessions, glms, roi_maskers, anat_roi_labels):
    """For each subject, session, condition, and ROI, computes the within-
    condition pattern stability (i.e., the average correlation between pairs
    of trials from the same condition)."""

    baseline_contrasts = {contrast_label: (conditions_plus, conditions_minus)
                          for contrast_label, (conditions_plus, conditions_minus)
                          in CONTRASTS.items() if len(conditions_minus) == 0}

    corr_dfs = []
    for subject, session, glm in zip(subjects, sessions, glms):

        for contrast_label, (conditions_plus, conditions_minus) in baseline_contrasts.items():

            if len(conditions_minus) > 0:

                warn(f'Skipping contrast {contrast_label} as it is not a ' +
                     'baseline contrast.')

                continue

            # else:
            #
            #     break

            design_cols = glm.design_matrices_[0].columns.tolist()

            assert len(conditions_plus) == 1
            all_conditions_plus = [col for col in design_cols
                                   if conditions_plus[0] in col]

            beta_imgs = []
            for condition_plus in all_conditions_plus:
                beta_img = compute_beta_img(glm,
                                            conditions_plus=(condition_plus,),
                                            conditions_minus=())
                beta_imgs.append(beta_img)

            pairs = list(combinations(beta_imgs, 2))

            condition = contrast_label.split('-')[1]
            assert condition in ['noise', 'pseudo', 'words']
            func_roi_labels = [roi_label for roi_label in roi_maskers.keys()
                               if f'-{condition}-' in roi_label]
            roi_labels = anat_roi_labels + func_roi_labels
            roi_maskers_ = {roi_label: roi_masker
                            for roi_label, roi_masker in roi_maskers.items()
                            if roi_label in roi_labels}

            for roi_label, roi_masker in roi_maskers_.items():

                betas = [roi_masker.transform(beta_img)[0]
                         for beta_img in beta_imgs]

                pairs = list(combinations(betas, 2))
                corrs = [np.corrcoef(pair)[0, 1] for pair in pairs]
                corr = np.mean(corrs)

                corr_df = pd.DataFrame({'subject': subject,
                                        'session': session,
                                        'contrast_label': contrast_label,
                                        'roi_label': roi_label,
                                        'r': corr},
                                       index=[0])
                corr_dfs.append(corr_df)

    return pd.concat(corr_dfs, ignore_index=True)


def run_stability_stats(corr_df):
    """Run linear mixed models on the correlation data, separately for each
    pair of condition and region of interest."""

    stat_dfs = []
    for contrast_label in corr_df['contrast_label'].unique():

        contrast_df = corr_df.query(f'contrast_label == "{contrast_label}"')

        for roi_label in contrast_df['roi_label'].unique():

            model_df = corr_df.query(f'contrast_label == "{contrast_label}" & ' +
                                     f'roi_label == "{roi_label}"')
            bs, zs = fit_mixed_model(FORMULA, model_df)
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
