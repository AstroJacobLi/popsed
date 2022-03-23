#!/bin/bash
#SBATCH -J PCA
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o pca.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_PCA.py

now=$(date +"%T")
echo "end time ... $now"
