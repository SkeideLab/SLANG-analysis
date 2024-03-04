from pathlib import Path

from bids import BIDSLayout

derivatives_dir = Path(__file__).parent.parent.parent.resolve()
output_dir = derivatives_dir / 'pybids'
task = 'language'

bids_dir = derivatives_dir.parent
layout = BIDSLayout(bids_dir)


sessions = layout.get_collections(task=task, level='session', types='scans',
                                  merge=True)
sessions_df = sessions.to_df()

output_filename = f'task-{task}_sessions.csv'
output_file = output_dir / output_filename
output_dir.mkdir(exist_ok=True, parents=True)
sessions_df.to_csv(output_file, index=False)
