#!/bin/sh

# Setup pytorch enviroment
module purge
module load anaconda3/2021.5
conda activate /scratch/gpfs/$USER/torch-env
module load rh/devtoolset/8
module load cudatoolkit/9.2