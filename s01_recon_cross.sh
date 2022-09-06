#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
remote=$2
participant_label=$3
session_label=$4
license_file=$5

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
datalad --on-failure ignore get --dataset . \
    sub-*/ses-*/*.json \
    sub-*/ses-*/*/*.json

# Copy license file so that it's available inside the container
job_license_file="freesurfer_license.txt"
cp "$license_file" "$job_license_file"

# Prepare output directory
freesurfer_dir="$job_dir/$deriv_name/freesurfer"
mkdir -p "$freesurfer_dir"

# Run surface reconstruction
output_dir="$freesurfer_dir/sub-${participant_label}_ses-$session_label"
datalad containers-run \
    --container-name "$deriv_name/code/containers/bids-freesurfer" \
    --input "sub-$participant_label/ses-$session_label/" \
    --output "$output_dir/" \
    --message "Cross-sectional surface reconstruction" \
    --explicit "\
$job_dir $freesurfer_dir participant \
--participant_label $participant_label \
--session_label $session_label \
--n_cpus $SLURM_CPUS_PER_TASK \
--steps cross-sectional \
--license_file $job_license_file \
--skip_bids_validator \
--3T true"

# Push large files to the RIA store
# Does not need a lock, no interaction with Git
datalad push --dataset "$deriv_name" --to output-storage

# Push to output branch
# Needs a lock to prevent concurrency issues
git -C "$deriv_name" remote add outputstore "$remote"
flock --verbose "$lockfile" git -C "$deriv_name" push outputstore

# Clean up everything
chmod -R +w "$job_dir"
rm -rf "$job_dir"

# And we're done
echo SUCCESS
