from itertools import combinations
from warnings import warn

import numpy as np
import pandas as pd
from bids import BIDSLayout
from similarity import ATLAS_FILE, get_roi_maskers
from univariate import (BIDS_DIR, CONTRASTS, DERIVATIVES_DIR, DF_QUERY,
                        FD_THRESHOLD, FMRIPREP_DIR, N_JOBS, PYBIDS_DIR, SPACE,
                        TASK, compute_beta_img, load_df, run_glms)

# Input parameters: Input/output directories
STABILITY_DIR = DERIVATIVES_DIR / 'stability'

# Input parameters: First-level GLM
BLOCKWISE = True
HRF_MODEL = 'glover'
SMOOTHING_FWHM = None
SAVE_RESIDUALS = False


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
    df = load_df(layout, TASK, percs_non_steady, percs_outliers, DF_QUERY)
    subjects = df['subject'].tolist()
    sessions = df['session'].tolist()
    good_ixs = list(df.index)

    # Exclude subjects/session that don't match the query
    glms = [glms[ix] for ix in good_ixs]
    mask_imgs = [mask_imgs[ix] for ix in good_ixs]

    # Get anatomical and functional regions of interest
    roi_maskers = get_roi_maskers(ATLAS_FILE, ref_img=mask_imgs[0])

    # Run pattern stability analysis
    corr_df = compute_stability(subjects, sessions, glms, roi_maskers)

    # Save results
    df = pd.merge(df, corr_df, on=['subject', 'session'])
    df_filename = f'task-{TASK}_space-{SPACE}_desc-stability.tsv'
    df_file = STABILITY_DIR / df_filename
    df.to_csv(df_file, sep='\t', index=False, float_format='%.4f')


def compute_stability(subjects, sessions, glms, roi_maskers):
    """For each subject, session, condition, and ROI, computes the within-
    condition pattern stability (i.e., the average correlation between pairs
    of trials from the same condition)."""

    corr_dfs = []
    for subject, session, glm in zip(subjects, sessions, glms):

        for contrast_label, (conditions_plus, conditions_minus) in CONTRASTS.items():

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

            for roi_label, roi_masker in roi_maskers.items():

                corrs = []
                for pair in pairs:

                    betas = [roi_masker.transform(beta_img)[0]
                             for beta_img in pair]
                    corr = np.corrcoef(betas)[0, 1]
                    corrs.append(corr)

                corr = np.mean(corrs)

                corr_df = pd.DataFrame({'subject': subject,
                                        'session': session,
                                        'contrast_label': contrast_label,
                                        'roi_label': roi_label,
                                        'r': corr},
                                       index=[0])
                corr_dfs.append(corr_df)

    return pd.concat(corr_dfs, ignore_index=True)
