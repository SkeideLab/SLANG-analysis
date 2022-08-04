from pathlib import Path
from os import environ

from simple_slurm import Slurm


def add_container(ds, name, url, bids_path='.'):
    """Stores a container (e.g., from Docker Hub) and in the dataset"""

    container_list = ds.containers_list()
    container_names = [c['name'] for c in container_list]
    if name not in container_names:
        call_fmt_list = [
            'singularity run',
            f'--bind {bids_path}:/data',
            f'--bind {bids_path}/.work:/work',
            '--bind $HOME/.cache/templateflow:/home/.cache/templateflow',
            '--cleanenv --home /home --no-home {img} {cmd}']
        call_fmt = ' '.join(call_fmt_list)
        ds.containers_add(name, url, call_fmt=call_fmt, update=True)


def get_templates(templates, bids_ds):
    """Downlaods standard templates form templateflow into the dataset
    Parameters
    ----------
    templates : list
        A list of template names (may include nonstandard templates).
    """

    # Initialize templateflow directory
    templateflow_dir = Path(bids_ds.path) / 'derivatives/templateflow'
    environ['TEMPLATEFLOW_HOME'] = str(templateflow_dir)
    import templateflow.api as tflow

    # Make sure the default templates needed for fmriprep are always present
    fmriprep_templates = ['OASIS30ANTs', 'MNI152NLin2009cAsym']
    all_templates = set(templates) | set(fmriprep_templates)

    # Check all *standard* templates, excluding nonstandard ones
    # See https://fmriprep.org/en/stable/spaces.html#nonstandard-spaces
    non_standard_templates = [
        'T1w', 'anat', 'fsnative', 'func', 'bold', 'run', 'boldref, sbref']
    for template in all_templates:
        if template not in non_standard_templates:

            # Get template name (excluding cohort, resolution, etc.)
            template = template.split(':')[0]

            # Actually download the template(s)
            _ = tflow.get(template, raise_empty=True)

    # Save changes
    bids_ds.save(
        templateflow_dir, message='Initialize/update templateflow templates')


def submit_job(args_list, cpus=8, mem=32000, time='24:00:00', log_dir='logs/',
               dependency_jobs=[], dependency_type='afterok', job_name='job'):
    """Submits a single batch job via SLURM, which can depend on other jobs.

    Parameters
    ----------
    args_list : list
        A list of shell commands and arguments. The first element will usually
        be the path of a shell script and the following elements the input
        arguments to this script.
    cpus : int, default=8
        The number of CPUs that the batch job should use.
    mem : int, default=320000
        The amount of memory (in MB) that the abtch job should use.
    time : str, default='24:00:00'
        The maximum run time (in format 'HH:MM:SS') that the batch job can use.
        Must exceed 24 hours.
    log_dir : str or Path, default='logs/'
        Directory to which the standard error and output messages of the batch
        job should be written.
    dependency_jobs : int or list, default=[]
        Other SLURM batch job IDs on which the current job depends. Can be used
        to create a pipeline of jobs that are executed after one another.
    dependency_type : str, default='afterok
        How to handle the 'dependency_jobs'. Must be one of ['after',
        'afterany', 'afternotok', 'afterok', 'singleton']. See [1] for further
        information. 
    job_dir : str
        Name of the batch job for creating meaningful log file names.

    Returns
    -------
    job_id : int
        The job ID of the submitted SLURM batch job.

    Notes
    -----
    [1] https://hpc.nih.gov/docs/job_dependencies.html
    """

    # Join arguments to a single bash command
    cmd = ' '.join(str(arg) for arg in args_list)

    # Create directory for output logs
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    error = f'{log_dir}/slurm-%j-{job_name}.out'
    output = f'{log_dir}/slurm-%j-{job_name}.out'

    # Prepare job scheduler
    slurm = Slurm(cpus_per_task=cpus, error=error, mem=mem, nodes=1, ntasks=1,
                  output=output, time=time, job_name=job_name)

    # Make the current job depend on previous jobs
    if dependency_jobs != []:
        if isinstance(dependency_jobs, int):
            dependency_jobs = [dependency_jobs]
        dependency_str = ':'.join([str(job_id) for job_id in dependency_jobs])
        dependency = {dependency_type: dependency_str}
        slurm.set_dependency(dependency)

    # Submit
    print('Submitting', cmd)
    job_id = slurm.sbatch(cmd)

    return job_id
