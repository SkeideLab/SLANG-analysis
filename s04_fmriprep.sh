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
git -C "$deriv_name" checkout -b "job-$SLURM_JOB_ID"

# Make sure that BIDS metadata is available
datalad --on-failure ignore get --dataset . \
  sub-*/ses-*/*.json \
  sub-*/ses-*/*/*.json

# Copy license file so that it's available inside the container
job_license_file="$job_dir/freesurfer_license.txt"
cp "$license_file" "$job_license_file"

# Make sure that fMRIPrep finds pre-downloaded templates
templateflow_dir="$job_dir/$deriv_name/templateflow"
datalad get --dataset "$deriv_name" "$templateflow_dir"
export SINGULARITYENV_TEMPLATEFLOW_HOME="$templateflow_dir"

# Create pyBIDS filter so that we can process one session at a time
filter_string='{"t1w":{"datatype":"anat","session":"'"$session"'","suffix":"T1w"},"bold":{"datatype":"func","session":"'"$session"'","suffix":"bold"}}'
filter_file="$job_dir/$deriv_name/bids_filter.json"
echo "$filter_string" >"$filter_file"

# Link pre-computed FreeSurfer data
datalad get "$deriv_name/freesurfer/sub-${participant}_ses-${session}"
fmriprep_dir="$job_dir/$deriv_name/fmriprep"
fs_subjects_dir="$fmriprep_dir/sourcedata/freesurfer"
mkdir -p "$fs_subjects_dir"
ln -s "../../../freesurfer/sub-${participant}_ses-${session}" \
  "$fs_subjects_dir/sub-${participant}"

# Make sure previous fmriprep data from this subject is available
datalad --on-failure ignore get "$fmriprep_dir/sub-$participant/figures"

# Start the job from the parent directory
# So that we can have the `work_dir` outside the `job_dir`
cd "$tmp_dir"
work_dir="$tmp_dir/work_job_$SLURM_JOB_ID"

# Run fmriprep
datalad containers-run \
  --container-name "$deriv_name/code/containers/bids-fmriprep" \
  --dataset "$job_dir" \
  --input "$job_dir/sub-$participant" \
  --input "$fs_subjects_dir" \
  --input "$job_dir/dataset_description.json" \
  --input "$filter_file" \
  --output "$fmriprep_dir/sub-${participant}" \
  --output "$fmriprep_dir/.bidsignore" \
  --output "$fmriprep_dir/dataset_description.json" \
  --output "$fmriprep_dir/desc-aparcaseg_dseg.tsv" \
  --output "$fmriprep_dir/desc-aseg_dseg.tsv" \
  --output "$fmriprep_dir/sub-${participant}.html" \
  --message "Preprocess functional data" \
  --explicit "\
$job_dir $fmriprep_dir participant \
--skip_bids_validation \
--bids-filter-file $filter_file \
--participant-label $participant \
--nprocs $SLURM_CPUS_PER_TASK \
--mem $SLURM_MEM_PER_NODE \
--output-spaces ${output_spaces[*]} sub$participant \
--random-seed 12345 \
--fd-spike-threshold $fd_thres \
--fs-license-file $job_license_file \
--fs-subjects-dir $fs_subjects_dir \
--work-dir $work_dir \
--stop-on-first-crash \
--notrack"

# Re-name QC report to prevent merge conflicts
old_html_file="$fmriprep_dir/sub-$participant.html"
new_html_file="$fmriprep_dir/sub-${participant}_ses-$session.html"
mv "$old_html_file" "$new_html_file"
datalad save \
  --dataset "$job_dir/$deriv_name" \
  --message "Rename fMRIprep report to include session info" \
  "$old_html_file" "$new_html_file"

# Push large files to the RIA store
# Does not need a lock, no interaction with Git
datalad push --dataset "$job_dir/$deriv_name" --to output-storage

# Push to output branch
# Needs a lock to prevent concurrency issues
git -C "$job_dir/$deriv_name" remote add outputstore "$remote"
flock --verbose "$lockfile" git -C "$job_dir/$deriv_name" push outputstore

# Clean up everything
chmod -R +w "$job_dir" "$work_dir"
rm -rf "$job_dir" "$work_dir"

# And we're done
echo SUCCESS
