#!/bin/bash
#SBATCH -J NDE_GAMA_MOCK_20_100_50_0.0003_40_20_12_0_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --export=ALL
#SBATCH --array=0-5
#SBATCH -o ./log/NDE_GAMA_MOCK_20_100_50_0.0003_40_20_12_0_5.%a.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_gama.py --multijobs=True --seed_low=0 --seed_high=5 --n_samples=10000 --num_transforms=20 --num_bins=50 --hidden_features=100 --output_dir=./NDE/GAMA/anneal/real/lr3e-4_ann12_zscore_40e/ --add_noise=True --max_lr=0.0003 --max_epochs=40 --anneal_coeff=20 --anneal_tau=12


now=$(date +"%T")
echo "end time ... $now"
