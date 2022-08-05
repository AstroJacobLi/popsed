#!/bin/bash
#SBATCH -J NDE_GAMA_DR3_0_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --array=0-5
#SBATCH -o ./log/NDE_GAMA_DR3_0_5_20_100_50.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_gama.py --num_transforms=20 --num_bins=50 --hidden_features=100 --output_dir=./NDE/GAMA/NMF/nde_theta_NMF_CDF_DR3_20trans/


now=$(date +"%T")
echo "end time ... $now"
