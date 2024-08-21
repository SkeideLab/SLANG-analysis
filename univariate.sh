#!/bin/bash -l
# Standard output and error:
#SBATCH -o /ptmp/aenge/SLANG/derivatives/code/logs/slurm-%j-univariate.out
#SBATCH -e /ptmp/aenge/SLANG/derivatives/code/logs/slurm-%j-univariate.out
# Initial working directory:
#SBATCH -D /ptmp/aenge/SLANG/derivatives/code
# Job Name:
#SBATCH -J univariate
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#
# Memory per node:
#SBATCH --mem 512000
#
# Wall clock limit:
#SBATCH --time=24:00:00

# singularity exec \
#     --bind /ptmp/aenge/SLANG:/ptmp/aenge/SLANG \
#     --cleanenv \
#     /ptmp/aenge/SLANG/derivatives/code/slang-analysis_latest.sif \
#     python3 /ptmp/aenge/SLANG/derivatives/code/univariate.py

singularity exec \
    --bind /ptmp/aenge/SLANG:/ptmp/aenge/SLANG \
    --cleanenv \
    /ptmp/aenge/SLANG/derivatives/code/slang-analysis_latest.sif \
    python3 /ptmp/aenge/SLANG/derivatives/code/univariate.py
