#!/bin/bash
#SBATCH -J NDE_NMF_15_20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o NDE_NMF_15_20.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde.py --seed_low=15 --seed_high=20 --output_dir=./NDE/NMF/nde_theta_NMF_sdss_noise_large/


now=$(date +"%T")
echo "end time ... $now"
