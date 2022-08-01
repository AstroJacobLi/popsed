#!/bin/bash
#SBATCH -J NDE_NSA_pretrained_20_25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o ./log/NDE_NSA_pretrained_20_25.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_cdf_nsa_pretrained.py --seed_low=20 --seed_high=25 --num_transforms=10 --num_bins=40 --hidden_features=100 --only_penalty=False --output_dir=./NDE/NMF/nde_theta_NMF_CDF_NSA_pretrained_blur0p1/


now=$(date +"%T")
echo "end time ... $now"
