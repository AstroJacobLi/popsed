#!/bin/bash
#SBATCH -J NDE_GAMA_DR3_5_6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --array=5-6
#SBATCH -o ./log/NDE_GAMA_DR3_5_6_10_50_50.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_gama.py --num_transforms=10 --num_bins=50 --hidden_features=50 --output_dir=./NDE/GAMA/NMF/nde_theta_NMF_CDF_DR3/


now=$(date +"%T")
echo "end time ... $now"
