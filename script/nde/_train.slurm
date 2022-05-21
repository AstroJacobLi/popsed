#!/bin/bash
#SBATCH -J NDE_NMF_NSA_1_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o NDE_NMF_NSA_1_5.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_cdf_nsa.py --seed_low=1 --seed_high=5 --num_transforms=10 --num_bins=40 --hidden_features=200 --only_penalty=False --output_dir=./NDE/NMF/nde_theta_NMF_CDF_NSA_given_redshift_blur/


now=$(date +"%T")
echo "end time ... $now"
