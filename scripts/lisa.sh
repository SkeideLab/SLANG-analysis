#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
remote=$2
participant=$3
session=$4
task=$5
space=$6
del_initial_volumes=$7
shift 7
contrasts_and_labels=("$@")

# Load Singularity for running containerized commands
module load singularity

# Create temporary location
tmp_dir="/ptmp/$USER/tmp"
mkdir -p "$tmp_dir"

# Clone the analysis dataset
# Flock makes sure that pull and push does not interfere with other jobs
bids_dir="$deriv_dir/.."
lockfile="$bids_dir/.git/datalad_lock"
job_dir="$tmp_dir/ds_job_$SLURM_JOB_ID"
flock --verbose "$lockfile" datalad clone "$bids_dir" "$job_dir"
cd "$job_dir"

# Announce the clone to be temporary
git submodule foreach --recursive git annex dead here

# Checkout a unique subdataset branch
deriv_name=$(basename "$deriv_dir")
datalad get --no-data "$deriv_name"
git -C "$deriv_name" checkout -b "job-$SLURM_JOB_ID"

# Make sure all JSON metadata is available
shopt -s globstar # For recursive `**` globbing syntax
datalad --on-failure ignore get --dataset . -- **/*.json

# Extract contrasts and correpsonding contrast labels
contrasts=("${contrasts_and_labels[@]::$((${#contrasts_and_labels[@]} / 2))}")
labels=("${contrasts_and_labels[@]:$((${#contrasts_and_labels[@]} / 2))}")

# Make sure container subdataset is available
datalad get --no-data "$deriv_name/code/containers"

# Run LISA
lisa_dir="$deriv_name/lisa"
datalad containers-run \
    --container-name "$deriv_name/lisa" \
    --input "sub-$participant/ses-$session" \
    --input "$deriv_name/fmriprep/sub-$participant/ses-$session" \
    --output "$lisa_dir/sub-$participant/ses-$session" \
    --message "Fit first-level GLM with LISA" \
    --explicit "\
$job_dir $lisa_dir participant \
--participant_label $participant \
--session_label $session \
--task_label $task \
--space_label $space \
--del_initial_volumes $del_initial_volumes \
--contrasts ${contrasts[*]} \
--contrast_labels ${labels[*]} \
--perm 5000"

# Push large files to the RIA store
# Does not need a lock, no interaction with Git
datalad push --dataset "$job_dir/$deriv_name" --to output-storage

# Push to output branch
# Needs a lock to prevent concurrency issues
git -C "$job_dir/$deriv_name" remote add outputstore "$remote"
flock --verbose "$lockfile" git -C "$job_dir/$deriv_name" push outputstore

# Clean up everything
chmod -R +w "$job_dir"
rm -rf "$job_dir"

# And we're done
echo SUCCESS
