import pandas as pd
from scipy.stats import norm
from univariate import (BIDS_DIR, DERIVATIVES_DIR, SPACE, TASK, UNIVARIATE_DIR,
                        fit_mixed_models)

# Input parameters: File paths
SOURCEDATA_DIR = BIDS_DIR / 'sourcedata'
BEHAVIOR_DIR = DERIVATIVES_DIR / 'behavior'

# Input parameters: Behavioral tests
TEST_LABELS = {
    'dali_1_rapid_picture_naming_seconds': 'DALI picture naming (s)',
    'dali_2a_word_reading_words': 'DALI word reading (# words)',
    'dali_2b_word_reading_seconds': 'DALI word reading (s)',
    'dali_3_rhyme': 'DALI rhyme (# pairs)',
    'dali_4_syllable_replacement': 'DALI phoneme replacement (# items)',
    'dali_5_semantic_fluency': 'DALI semantic fluency (# words)',
    'dali_6_verbal_fluency': 'DALI verbal fluency (# words)',
    'dali_7a_nonword_reading_correct': 'DALI nonword reading (# correct)',
    'dali_7b_nonword_reading_seconds': 'DALI nonword reading (s)',
    'dali_8a_reading_comprehension_seconds': 'DALI comprehension (s)',
    'dali_8b_reading_comprehension_questions': 'DALI comprehension (# questions)',
    'dali_9_dictation': 'DALI dictation (# correct)',
    'wrat5_1_oral_math': 'WRAT oral math (# correct)',
    'wrat5_2_math_computation': 'WRAT computation (# correct)',
    'corsi_block_fwd': 'WM forward (# items)',
    'corsi_block_bwd': 'WM backward (# items)',
    'digit_span_fwd': 'Digit span forward (# digits)',
    'digit_span_bwd': 'Digit span test backward (# digits)',
    'ravens_cpm': 'Raven\'s CPM (# correct)'}

# Input parameters: Linear mixed models
FORMULA = 'score ~ time + (time | subject)'


def main():
    """Main function for running the full behavior analysis."""

    meta_file = UNIVARIATE_DIR / f'task-{TASK}_space-{SPACE}_desc-metadata.tsv'
    meta_df = pd.read_csv(meta_file, sep='\t',
                          dtype={'subject': str, 'session': str})
    meta_df = meta_df.query('include')

    behavior_file = SOURCEDATA_DIR / 'behavior.tsv'
    df = pd.read_csv(behavior_file, sep='\t', dtype={'session': str})

    df = df.drop(columns=['yyyy_mm_dd', 'comments'])
    df = pd.melt(df, id_vars=['subject', 'session'],
                 var_name='test', value_name='score')

    df = df.dropna()
    df = df.drop_duplicates(subset=['subject', 'session', 'test'], keep='last')

    meta_df_short = meta_df[['subject', 'session', 'time']]
    df = pd.merge(meta_df_short, df, on=['subject', 'session'], how='left')

    df['test'] = df['test'].\
        replace(TEST_LABELS).\
        astype('category').\
        cat.set_categories(TEST_LABELS.values())
    df = df.dropna()

    BEHAVIOR_DIR.mkdir(exist_ok=True, parents=True)

    df_filename = f'task-{TASK}_space-{SPACE}_desc-behavior_scores.tsv'
    df_file = BEHAVIOR_DIR / df_filename
    df.to_csv(df_file, sep='\t', index=False)

    labels, dfs = zip(*df.groupby('test'))

    res = fit_mixed_models(FORMULA, dfs)
    bs, zs = zip(*res)
    b0s, b1s = zip(*bs)
    z0s, z1s = zip(*zs)

    stat_df = pd.DataFrame({'test': labels,
                            'b0': b0s,
                            'b1': b1s,
                            'z0': z0s,
                            'z1': z1s,
                            'p0': norm.sf(z0s),
                            'p1': norm.sf(z1s)})

    stat_df_filename = f'task-{TASK}_space-{SPACE}_desc-behavior_stats.tsv'
    stat_df_file = BEHAVIOR_DIR / stat_df_filename
    stat_df.to_csv(stat_df_file, sep='\t', index=False, float_format='%.4f')


if __name__ == '__main__':
    main()
