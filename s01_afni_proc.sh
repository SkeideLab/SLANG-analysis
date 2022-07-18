#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
remote=$2
participant_label=$3
session_label=$4
task_label=$5
non_stationary_trs=$6
blur_size=$7
regress_censor_motion=$8
gltsym_contrast=$9

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

# Make sure that BIDS metadata is available
datalad get -d . \
    sub-*/ses-*/*.json \
    sub-*/ses-*/*/*.json \
    code/qc/sub-*/ses-*/*/*.json

# Run afni_proc
afni_dir="$deriv_name/afni"
datalad containers-run \
    --container-name "$deriv_name/bids-app-afni-proc" \
    --input "sub-$participant_label" \
    --output "$afni_dir" \
    --message "Preprocess functional data" \
    --explicit "\
/data /data/$afni_dir participant \
--participant_label $participant_label \
--session_label $session_label \
--task_label $task_label \
--non_stationary_trs $non_stationary_trs \
--blur_size $blur_size \
--gltsym_contrasts $gltsym_contrast \
--regress_censor_motion $regress_censor_motion"

# Push large files to the RIA store
# Does not need a lock, no interaction with Git
datalad push --dataset "$job_dir/$deriv_name" --to output-storage

# Push to output branch
# Needs a lock to prevent concurrency issues
git -C "$job_dir/$deriv_name" remote add outputstore "$remote"
flock --verbose "$lockfile" git -C "$job_dir/$deriv_name" push outputstore

# Clean up everything
rm -rf "$job_dir"

# And we're done
echo SUCCESS
