import json
from pathlib import Path

from datalad.api import Dataset

from helpers import get_templates, submit_job

# Read study-specific inputs from `run_params.json`
with open(f'{Path(__file__).parent.resolve()}/run_params.json', 'r') as fp:
    run_params = json.load(fp)

# Find derivatives Dataset
deriv_dir = Path(__file__).parent.parent.resolve()
deriv_ds = Dataset(deriv_dir)
deriv_name = deriv_dir.name

# Find BIDS Dataset
bids_dir = deriv_dir.parent
bids_ds = Dataset(bids_dir)

# Define directory for SLURM batch job log files
log_dir = deriv_dir / 'code' / 'logs'

# Create outputstore to store intermediate results from batch jobs
ria_dir = bids_dir / '.outputstore'
if not ria_dir.exists():
    ria_url = f'ria+file://{ria_dir}'
    deriv_ds.create_sibling_ria(
        ria_url, name='output', alias='derivatives', new_store_ok=True)
    deriv_ds.push(to='output')

# Find path of the dataset in the outputstore
remote = deriv_ds.siblings(name='output')[0]['url']

# Make sure that containers are available for the batch jobs
code_dir_name = Path(__file__).parent.name
containers_path = code_dir_name + '/containers/images/'
containers_dict = {
    'freesurfer': containers_path + 'bids/bids-freesurfer--6.0.1-6.1.sing',
    'fmriprep': containers_path + 'bids/bids-fmriprep--21.0.2.sing'}
deriv_ds.get(containers_dict.values())

# Download standard templates so they are available for the batch jobs
output_spaces = run_params['output_spaces']
get_templates(output_spaces, bids_ds)

# Make sure the output store is up to date
# Otherwise there might be an error when pushing results back from the
# batch jobs
_ = deriv_ds.push(to='output')
_ = deriv_ds.push(to='output-storage')

# # An additional garbage collection in the outputstore might also be useful
# # but can take a couple of minutes (this requires the gitpython package)
# from git import Repo
# _ = Repo(remote).git.gc()

# Extract participant and session labels from directory structure
participant_session_dirs = list(bids_dir.glob('sub-*/ses-*/'))
participants_sessions = [
    (d.parent.name.replace('sub-', ''), d.name.replace('ses-', ''))
    for d in participant_session_dirs]
participants_sessions.sort()

# # Select a subset of participants/sessions for debugging
# participants_sessions = [('SA27', '01')]

# Cross-sectional surface reconstruction
script = f'{deriv_dir}/code/s01_recon_cross.sh'
license_file = run_params['license_file']
job_ids = []
for participant, session in participants_sessions:
    args = [script, deriv_dir, remote, participant, session, license_file]
    this_job_id = submit_job(
        args, job_name=f's01_recon_cross_sub-{participant}_ses-{session}',
        log_dir=log_dir)
    job_ids.append(this_job_id)

# Merge branches back into the dataset once they've finished
script = f'{deriv_dir}/code/s02_merge.sh'
pipeline_dir = deriv_dir / 'freesurfer'
pipeline_description = 'FreeSurfer'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, dependency_jobs=job_ids, dependency_type='afterany',
                    log_dir=log_dir, job_name='s02_merge')

# Compute subject-level template
script = f'{deriv_dir}/code/s03_recon_template.sh'
license_file = run_params['license_file']
job_ids = []
participants = list(set([elem[0] for elem in participants_sessions]))
for participant in participants:
    args = [script, deriv_dir, remote, participant, license_file]
    this_job_id = submit_job(
        args, dependency_jobs=job_id, log_dir=log_dir,
        job_name=f's03_recon_template_sub-{participant}')
    job_ids.append(this_job_id)

# Merge branches back into the dataset once they've finished
script = f'{deriv_dir}/code/s02_merge.sh'
pipeline_dir = deriv_dir / 'freesurfer'
pipeline_description = 'FreeSurfer'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, dependency_jobs=job_ids, dependency_type='afterany',
                    log_dir=log_dir, job_name='s02_merge')

# Preprocessing with fmriprep
script = f'{deriv_dir}/code/s04_fmriprep.sh'
license_file = run_params['license_file']
fd_thres = run_params['fd_thres']
job_ids = []
for participant, session in participants_sessions:
    args = [script, deriv_dir, remote, participant, session, license_file,
            fd_thres, *output_spaces]
    this_job_id = submit_job(
        args, dependency_jobs=job_id, log_dir=log_dir,
        job_name=f's04_fmriprep_sub-{participant}_ses-{session}')
    job_ids.append(this_job_id)

# Merge branches back into the dataset once they've finished
script = f'{deriv_dir}/code/s02_merge.sh'
pipeline_dir = deriv_dir / 'fmriprep'
pipeline_description = 'fMRIPrep'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, dependency_jobs=job_ids, dependency_type='afterany',
                    log_dir=log_dir, job_name='s02_merge')
