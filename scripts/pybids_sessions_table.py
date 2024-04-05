from pathlib import Path

import pandas as pd
from bids import BIDSLayout

derivatives_dir = Path(__file__).parent.parent.parent.resolve()
output_dir = derivatives_dir / 'pybids'
task = 'language'
fd_threshs = [0.2, 0.5, 1.0, 2.4]


def get_fd_num(confounds_file, fd_thresh):

    confounds = pd.read_csv(confounds_file, sep='\t')
    fd = confounds['framewise_displacement']
    fd_num = fd[fd > fd_thresh].count()

    return fd_num


bids_dir = derivatives_dir.parent
fmriprep_dir = derivatives_dir / 'fmriprep'
layout = BIDSLayout(bids_dir, derivatives=fmriprep_dir)

sessions = layout.get_collections(task=task, level='session', types='scans',
                                  merge=True)
sessions_df = sessions.to_df()

confounds_files = sorted(list(layout.get('filename', task=task,
                                         desc='confounds', suffix='timeseries',
                                         extension='tsv')))
assert len(confounds_files) == len(sessions_df)

for fd_thresh in fd_threshs:
    fd_nums = [get_fd_num(confounds_file, fd_thresh)
               for confounds_file in confounds_files]
    fd_thresh_underscore = str(fd_thresh).replace('.', '_')
    sessions_df[f'fd_num_greater_{fd_thresh_underscore}'] = fd_nums

output_filename = f'task-{task}_sessions.tsv'
output_file = output_dir / output_filename
output_dir.mkdir(exist_ok=True, parents=True)
sessions_df.to_csv(output_file, sep='\t', index=False)
