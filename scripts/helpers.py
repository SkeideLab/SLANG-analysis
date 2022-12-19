from os import environ
from pathlib import Path

from simple_slurm import Slurm


def get_templates(templates, bids_ds, deriv_name):
    """Downloads standard brain templates from TemplateFlow[1] to the dataset.

    Parameters
    ----------
    templates : list of str
        The names of the desired templates. See the TemplateFlow website[1]
        for the templates that are available. May include additional
        non-standard templates.
    bids_ds : datalad.api.Dataset
        The BIDS dataset to which the downloaded templates will be saved.
    deriv_name : str
        The name of the derivatives subdataset inside the BIDS dataset.

    Notes
    -----
    [1] https://www.templateflow.org
    """

    templateflow_dir = Path(bids_ds.path) / deriv_name / 'templateflow'
    environ['TEMPLATEFLOW_HOME'] = str(templateflow_dir)
    import templateflow.api as tflow

    # These two templates will *always* be required by fMRIPrep
    fmriprep_templates = ['OASIS30ANTs', 'MNI152NLin2009cAsym']
    templates = set(templates) | set(fmriprep_templates)

    # These templates are non-standard and don't need to be downloaded
    # See https://fmriprep.org/en/stable/spaces.html#nonstandard-spaces
    non_standard_templates = [
        'T1w', 'anat', 'fsnative', 'func', 'bold', 'run', 'boldref, sbref']
    templates -= set(non_standard_templates)

    for template in templates:
        template = template.split(':')[0]  # Exclude additional modifiers
        _ = tflow.get(template, raise_empty=True)

    bids_ds.save(
        templateflow_dir, message='Initialize/update templateflow templates')


def get_ria_remote(ds, ria_dir, sibling_name='output'):
    """Creates a RIA store[1] dataset sibling if necessary and returns its URL.

    Parameters
    ----------
    ds : datalad.api.Dataset
        The main (BIDS/derivatives) dataset for which the RIA store sibling
        will be created.
    ria_dir : str or Path
        The desired directory path for the RIA store.
    sibling_name : str
        The name under which the RIA store sibling will be configured in the
        dataset.

    Returns
    -------
    remote : str
        The path/URL of the configered RIA store.

    Notes
    -----
    [1] https://handbook.datalad.org/en/latest/beyond_basics/101-147-riastores.html
    """

    siblings = [sib['name'] for sib in ds.siblings()]
    if not sibling_name in siblings:
        ria_url = f'ria+file://{ria_dir}'
        ds.create_sibling_ria(ria_url, name=sibling_name,
                              alias='derivatives', new_store_ok=True)

    return ds.siblings(name='output')[0]['url']


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

    cmd = ' '.join(str(arg) for arg in args_list)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    error = f'{log_dir}/slurm-%j-{job_name}.out'
    output = f'{log_dir}/slurm-%j-{job_name}.out'

    slurm = Slurm(cpus_per_task=cpus, error=error, mem=mem, nodes=1, ntasks=1,
                  output=output, time=time, job_name=job_name)

    if dependency_jobs != []:
        if isinstance(dependency_jobs, int):
            dependency_jobs = [dependency_jobs]
        dependency_str = ':'.join([str(job_id) for job_id in dependency_jobs])
        dependency = {dependency_type: dependency_str}
        slurm.set_dependency(dependency)

    print('Submitting', cmd)
    job_id = slurm.sbatch(cmd)

    return job_id
