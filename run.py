# %% [markdown]
# # fMRIPrep pipeline
#
# Welcome to our fMRI preprocessing pipeline using [fMRIPrep][1]! This Python
# script will guide you to all the steps that you need for preprocessing your
# data. It will work for both cross-sectional and longitudinal data, as long
# as they are in BIDS format with subject- (and optionally session-)specific
# sub-directories.
#
# ## Load modules and helper functions

# %%
import json
from pathlib import Path

from datalad.api import Dataset
from scripts.helpers import get_ria_remote, get_templates, submit_job

# %% [markdown]
# ## Find DataLad datasets
#
# We automatically detect the **BIDS derivatives dataset** (where preprocessing
# outputs are to be stored) based on the location of the current file. We also
# detect the **BIDS dataset** as the parent directory of the derivatives
# dataset. Finally, we load the study-specific preprocessing parameters that
# are specified in the [`run_params.json`](run_params.json) file.

# %%
code_dir = Path(__file__).parent.resolve()
log_dir = code_dir / 'logs'
script_dir = code_dir / 'scripts'

deriv_dir = code_dir.parent
deriv_ds = Dataset(deriv_dir)
deriv_name = deriv_dir.name

bids_dir = deriv_dir.parent
bids_ds = Dataset(bids_dir)

with open(code_dir / 'run_params.json', 'r') as fp:
    run_params = json.load(fp)

# %% [markdown]
# ## Download containers
#
# The software tools for preprocessing are provided as [Singularity
# containers][2]. Since the batch jobs don't have access to the internet, we
# need to download the relevant containers beforehand. This is done via the
# [ReproNim containers subdataset][3].

# %%
containers_prefix = f'{code_dir.name}/containers/images'
containers_dict = {
    'fmriprep': f'{containers_prefix}/bids/bids-fmriprep--24.0.1.sing'}
_ = deriv_ds.get(containers_dict.values())

# %% [markdown]
# ## Install custom container for statistical analysis

# %%
installed_containers = [elem['name'] for elem in
                        bids_ds.containers_list(result_renderer='disabled')]

if not 'python-julia-afni' in installed_containers:
    bind_dir = Path(str(bids_dir).replace('/raven', ''))
    bids_ds.containers_add(
        name='python-julia-afni',
        url='docker://skeidelab/python-julia-afni:0.2',
        call_fmt=f'singularity exec --bind {bind_dir}:{bind_dir} ' +
            '--cleanenv {img} {cmd}')

# %% [markdown]
# ## Download templates
#
# Like the software containers, any standard brain templates that are needed
# during preprocessing by fMRIPrep need to be pre-downloaded from
# [TemplateFlow][4] so that they are available to the batch jobs.

# %%
output_spaces = run_params['output_spaces']
get_templates(output_spaces, bids_ds, deriv_name)

# %% [markdown]
# ## Create or find output store
#
# The [reproducible DataLad workflow][5] that we are using requires an
# intermediate copy of the derivatives dataset, which is used for pushing the
# results from individual batch jobs (typically preprocessing data from a
# single participant) into the same dataset. This intermediate dataset is
# called the **output store** and is in [RIA format][6]. Here we create this
# RIA output store (if it doesn't exist) and retrieve it's address so that we
# can pass it to the batch jobs. We also make sure that the output store is up
# to date with the the main derivatives dataset by pushing to it.

# %%
ria_dir = bids_dir / '.outputstore'
remote = get_ria_remote(deriv_ds, ria_dir)

# _ = deriv_ds.push(to='output')
# _ = deriv_ds.push(to='output-storage')
# _ = Repo(remote).git.gc() # Takes a couple of minutes, usually not necessary

# %% [markdown]
# ## Extract participant labels
#
# Here we retrieve all the available participant labels based on the BIDS
# directory structure. If you only want to process a subset of subjects, you
# can overwrite the `participants` variable with a cusomt list of subjects.

# %%
participant_dirs = list(bids_dir.glob('sub-*/'))
participants = sorted([d.name.replace('sub-', '') for d in participant_dirs])
# participants = ['SA27']  # Custom selection for debugging

# %% [markdown]
# ## Run fMRIPrep
#
# Here we actually run the preprocessing with fMRIPrep. Each subject
# (with all its sessions) is processed in a separate batch job, submitted via
# the `sbatch` command in the SLURM scheduling system. For the details of what
# happens inside each batch job and how we parameterize fMRIPrep, please check
# out the shell script that is referenced below (it's in the same directiry as
# this script).

# %%
script = script_dir / 'fmriprep.sh'
license_file = run_params['license_file']
fd_thres = run_params['fd_thres']
job_ids = []
for participant in participants:
    args = [script, deriv_dir, remote, participant,
            license_file, fd_thres, *output_spaces]
    job_name = f'fmriprep_sub-{participant}'
    this_job_id = submit_job(args, log_dir=log_dir, job_name=job_name)
    job_ids.append(this_job_id)

# %% [markdown]
# ## Merge job branches
#
# After each preprocessing step, we merge the job-specific branches with the
# preprocessed results from the output store back into the main derivatives
# dataset. This is implemented via another batch job that depends on all
# participant-specific jobs from the last step having finished.

# %%
script = script_dir / 'merge.sh'
pipeline_dir = deriv_dir / 'fmriprep'
pipeline_description = 'fMRIPrep'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, dependency_jobs=job_ids, dependency_type='afterany',
                    log_dir=log_dir, job_name='merge')

# %%
script = code_dir / 'univariate.sh'
args = [script, deriv_dir]
job_id = submit_job(args, cpus=72, mem=512000, dependency_jobs=job_id,
                    dependency_type='afterok', log_dir=log_dir,
                    job_name='univariate')

# %% [markdown]
# [1]: https://fmriprep.org/en/stable/index.html
# [2]: http://handbook.datalad.org/en/latest/basics/101-133-containersrun.html
# [3]: https://github.com/ReproNim/containers
# [4]: https://www.templateflow.org
# [5]: https://doi.org/10.1038/s41597-022-01163-2
# [6]: https://handbook.datalad.org/en/latest/beyond_basics/101-147-riastores.html
# [7]: https://doi.org/10.1038/s41467-018-06304-z
