#!/bin/sh

# Setup pytorch enviroment
module purge
module load anaconda3/2021.5
conda activate /scratch/gpfs/$USER/torch-env