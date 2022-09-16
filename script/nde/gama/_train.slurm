#!/bin/bash
#SBATCH -J NDE_GAMA_MOCK_0_20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=22G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

#SBATCH -o ./log/NDE_GAMA_MOCK_0_20_20_128_50.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_mock.py --seed_low=0 --seed_high=20 --n_samples=10000 --num_transforms=20 --num_bins=50 --hidden_features=128 --output_dir=./NDE/GAMA/NMF/nde_theta_NMF_CDF_mock_flatdirich_nonoise/ --add_noise=False --smallblur=True


now=$(date +"%T")
echo "end time ... $now"
