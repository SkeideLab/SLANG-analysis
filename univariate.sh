#!/bin/bash -l
# Standard output and error:
#SBATCH -o /ptmp/aenge/slang/data/derivatives/code/logs/mixedmodels.%j
#SBATCH -e /ptmp/aenge/slang/data/derivatives/code/logs/mixedmodels.%j
# Initial working directory:
#SBATCH -D /ptmp/aenge/slang/data/derivatives/code
# Job Name:
#SBATCH -J mixedmodels
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# Memory per node:
#SBATCH --mem 64000
#
# Wall clock limit:
#SBATCH --time=24:00:00

conda activate slang

python3 /ptmp/aenge/slang/data/derivatives/code/mixed_models_clustering.py
