#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
remote=$2
participant=$3
session=$4
license_file=$5
fd_thres=$6
shift 6
output_spaces=("$@")

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
# datalad get --no-data "$deriv_name/.containers"
git -C "$deriv_name" checkout -b "job-$SLURM_JOB_ID"

# Make sure that BIDS metadata is available
datalad get -d . \
  sub-*/ses-*/*.json \
  sub-*/ses-*/*/*.json \
  code/qc/sub-*/ses-*/*/*.json

# Link pre-computed FreeSurfer data
datalad get "$deriv_name/freesurfer/sub-${participant}_ses-${session}"
fmriprep_dir="$deriv_name/fmriprep"
fs_subjects_dir="$fmriprep_dir/sourcedata/freesurfer"
mkdir -p "$fs_subjects_dir"
ln -s "../../../freesurfer/sub-${participant}_ses-${session}" \
  "$fs_subjects_dir/sub-${participant}"

# Make sure previous fmriprep data from this subject is available
datalad get "$fmriprep_dir/sub-$participant/figures"

# Create pyBIDS filter so that we can process one session at a time
filter_string='{"t1w":{"datatype":"anat","session":"'"$session"'","suffix":"T1w"},"bold":{"datatype":"func","session":"'"$session"'","suffix":"bold"}}'
filter_file="$deriv_name/code/bids_filter.json"
echo "$filter_string" >"$filter_file"

# Copy license file into the job directory
job_license_file="$deriv_name/code/license.txt"
cp "$license_file" "$job_license_file"

# Create working directory
mkdir "$job_dir/.work"

# Run fmriprep
datalad containers-run \
  --container-name "$deriv_name/fmriprep" \
  --input "sub-$participant" \
  --input "$fs_subjects_dir" \
  --input "dataset_description.json" \
  --input "$filter_file" \
  --output "$fmriprep_dir/sub-${participant}" \
  --output "$fmriprep_dir/.bidsignore" \
  --output "$fmriprep_dir/dataset_description.json" \
  --output "$fmriprep_dir/desc-aparcaseg_dseg.tsv" \
  --output "$fmriprep_dir/desc-aseg_dseg.tsv" \
  --output "$fmriprep_dir/sub-${participant}.html" \
  --message "Preprocess functional data" \
  --explicit "\
/data /data/$fmriprep_dir participant \
--skip_bids_validation \
--bids-filter-file /data/$filter_file \
--participant-label $participant \
--nprocs $SLURM_CPUS_PER_TASK \
--mem $SLURM_MEM_PER_NODE \
--longitudinal \
--output-spaces ${output_spaces[*]} \
--random-seed 12345 \
--fd-spike-threshold $fd_thres \
--fs-license-file /data/$job_license_file \
--fs-subjects-dir /data/$fs_subjects_dir \
--work-dir /work \
--stop-on-first-crash \
--notrack"

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
