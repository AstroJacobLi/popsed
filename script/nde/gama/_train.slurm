#!/bin/bash
#SBATCH -J NDE_GAMA_DR3_0_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --export=ALL

#SBATCH -o ./log/NDE_GAMA_DR3_0_5_15_100_50.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_gama.py --seed_low=0 --seed_high=10 --num_transforms=15 --num_bins=50 --hidden_features=100 --output_dir=./NDE/GAMA/NMF/nde_theta_NMF_CDF_DR3/


now=$(date +"%T")
echo "end time ... $now"
