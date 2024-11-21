#!/bin/bash -l

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1

# Load Apptainer for running containerized commands
module load apptainer

# Activate conda environment
module load anaconda/3/2023.03
conda activate SLANG

# Go into the BIDS dataset
bids_dir="$deriv_dir/.."
cd "$bids_dir"

# Run the univariate analysis
datalad containers-run \
  --container-name "python-julia-afni" \
  --dataset "$bids_dir" \
  --input "sub-*" \
  --input "$deriv_dir/behavior" \
  --input "$deriv_dir/similarity" \
  --input "$deriv_dir/stability" \
  --input "$deriv_dir/univariate" \
  --output "$deriv_dir/behavior/figures" \
  --output "$deriv_dir/similarity/figures" \
  --output "$deriv_dir/stability/figures" \
  --output "$deriv_dir/univariate/figures" \
  --message "Run plotting" \
  --explicit "\
python3 $deriv_dir/code/scripts/plotting.py"

# And we're done
echo SUCCESS
