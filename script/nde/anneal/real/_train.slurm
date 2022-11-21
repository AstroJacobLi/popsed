#!/bin/bash
#SBATCH -J NDE_DR3_SNR1_20_100_60_0.0003_40_20_10_40_50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=15:59:59
#SBATCH --export=ALL

#SBATCH -o ./log/NDE_DR3_SNR1_20_100_60_0.0003_40_20_10_40_50.%a.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_gama.py --multijobs=False --seed_low=40 --seed_high=50 --n_samples=10000 --num_transforms=20 --num_bins=60 --hidden_features=100 --output_dir=./NDE/GAMA/anneal/real/lr3e-4_noisefree_40e/ --add_noise=False --max_lr=0.0003 --max_epochs=40 --anneal_coeff=20 --anneal_tau=10


now=$(date +"%T")
echo "end time ... $now"
