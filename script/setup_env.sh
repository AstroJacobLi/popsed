#!/bin/sh

# Setup pytorch enviroment
module purge
module load anaconda3/2022.5
conda activate /scratch/gpfs/$USER/torch-env
module load cudatoolkit/10.2