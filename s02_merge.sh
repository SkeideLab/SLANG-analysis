#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Parse arguments from the job scheduler as variables
deriv_dir=$1
pipeline_dir=$2
pipeline_description=$3
shift 3
merge_job_ids=("$@")

# Go into the dataset directory
cd "$deriv_dir"

# Create empty strings
local_branches=""
remote_branches=""
unsucessful_jobs=""

# Check which jobs finished succesfully
for job_id in "${merge_job_ids[@]}"; do
    log_file="$deriv_dir/code/logs/slurm-$job_id-*.out"
    if grep -Fxq "SUCCESS" $log_file; then
        local_branches+=" output/job-$job_id"
        remote_branches+=" job-$job_id"
    else
        unsucessful_jobs+=" $job_id"
    fi
done

# Python command to update JSON file with dataset description
# Since downstram programs (e.g., pyBIDS) will expect this field
description_file="$pipeline_dir/dataset_description.json"
PYCMD=$(
    cat <<EOF
import json
try:
    f = open('$description_file')
    data = json.load(f)
    f.close()
except:
    data = {}
data['PipelineDescription'] = {'Name': '$pipeline_description'}
json_object = json.dumps(data, indent=4)
f = open('$description_file', mode='w+')
f.write(json_object)
f.close()
EOF
)

# Merge and delete successful branches
if [ -n "$local_branches" ]; then
    datalad update --sibling output
    git merge -m "Merge batch job results" $local_branches
    git annex fsck --fast -f output-storage
    datalad get .
    git push --delete output $remote_branches

    # Save changes in BIDS superdataset
    datalad save -m "Update derivatives" -d .. .

    # Update dataset description file
    if test -f "$description_file"; then
        datalad unlock "$description_file"
    fi
    python3 -c "$PYCMD"
    datalad save -m "Update dataset description" -d .. "$description_file"

fi

# Warn about unsucessful branches
if [ -n "$unsucessful_jobs" ]; then
    echo "WARNING: Not merging unsuccessful batch jobs $unsucessful_jobs." \
        "Please check their log files and Dataset clones."
fi

# And we're done
echo SUCCESS
