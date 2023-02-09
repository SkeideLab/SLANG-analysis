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
code_dir_name = Path(__file__).parent.name
containers_path = code_dir_name + '/containers/images/'
containers_dict = {
    'fmriprep': containers_path + 'bids/bids-fmriprep--21.0.2.sing'}
_ = deriv_ds.get(containers_dict.values())

# %% [markdown]
# ## Add custom LISA container

# %%
deriv_ds.containers_add(
    name='lisa',
    url='docker://skeidelab/bids-app-lisa:latest',
    call_fmt='{img_dspath}/code/containers/scripts/singularity_cmd run {img} {cmd}',
    # update=True, # Uncomment if you want to update the container
    on_failure='ignore')
bids_ds.save('derivatives/.datalad/environments',
             message='Add/update containers')

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

_ = deriv_ds.push(to='output')
_ = deriv_ds.push(to='output-storage')
# _ = Repo(remote).git.gc() # Takes a couple of minutes, usually not necessary

# %% [markdown]
# ## Extract participant labels
#
# Here we retrieve all the available participant labels based on the BIDS
# directory structure. If you only want to process a subset of subjects, you
# can overwrite the `participants` variable with a cusomt list of subjects.

# %%
participant_session_dirs = sorted(list(bids_dir.glob('sub-*/ses-*/')))
participants_sessions = [(d.parent.name.replace('sub-', ''),
                          d.name.replace('ses-', ''))
                         for d in participant_session_dirs]
# participants_sessions = [('SA31', '01')]  # Custom selection for debugging
participants = [p_s[0] for p_s in participants_sessions]

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
script = code_dir / 'scripts/fmriprep.sh'
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
script = code_dir / 'scripts/merge.sh'
pipeline_dir = deriv_dir / 'fmriprep'
pipeline_description = 'fMRIPrep'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, dependency_jobs=job_ids, dependency_type='afterany',
                    log_dir=log_dir, job_name='merge')

# %% [markdown]
# ## Run first level GLM with LISA
#
# To do: Describe [LISA][7]

# %%
script = code_dir / 'scripts/lisa.sh'

# To do: Pass via `run_params.json`
task = 'language'
space = 'T1w'
smooth_fwhm = 4.8
fd_thres = run_params['fd_thres']
contrasts = [
    '+audios_noise+audios_pseudo+audios_words',
    '+images_noise+images_pseudo+images_words',
    '+audios_noise+audios_pseudo+audios_words-images_noise-images_pseudo-images_words',
    '+images_noise+images_pseudo+images_words-audios_noise-audios_pseudo-audios_words',
    '+audios_words-audios_pseudo', '+audios_pseudo-audios_noise',
    '+images_words-images_pseudo', '+images_pseudo-images_noise']
contrast_labels = ['audios-minus-null',
                   'images-minus-null',
                   'audios-minus-images',
                   'images-minus-audios',
                   'audios-words-minus-pseudo',
                   'audios-pseudo-minus-noise',
                   'images-words-minus-pseudo',
                   'images-pseudo-minus-noise']
contrasts_and_labels = contrasts + contrast_labels

job_ids = []
for participant, session in participants_sessions:
    args = [script, deriv_dir, remote, participant, session, task, space,
            smooth_fwhm, fd_thres, *contrasts_and_labels]
    job_name = f'lisa_sub-{participant}_ses-{session}'
    this_job_id = submit_job(args, cpus=40, mem=185000, log_dir=log_dir,
                             dependency_jobs=job_id, job_name=job_name)
    job_ids.append(this_job_id)

# %% [markdown]
# ## Merge job branches

# %%
script = code_dir / 'scripts/merge.sh'
pipeline_dir = deriv_dir / 'lisa'
pipeline_description = 'LISA'
args = [script, deriv_dir, pipeline_dir, pipeline_description, *job_ids]
job_id = submit_job(args, cpus=40, mem=185000, dependency_jobs=job_ids, 
                    dependency_type='afterany', log_dir=log_dir,
                    job_name='merge')

# %% [markdown]
# [1]: https://fmriprep.org/en/stable/index.html
# [2]: http://handbook.datalad.org/en/latest/basics/101-133-containersrun.html
# [3]: https://github.com/ReproNim/containers
# [4]: https://www.templateflow.org
# [5]: https://doi.org/10.1038/s41597-022-01163-2
# [6]: https://handbook.datalad.org/en/latest/beyond_basics/101-147-riastores.html
# [7]: https://doi.org/10.1038/s41467-018-06304-z
