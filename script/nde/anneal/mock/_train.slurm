#!/bin/bash
#SBATCH -J NDE_GAMA_MOCK_15_60_50_0.0003_40_20_12_0_3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=1:30:00
#SBATCH --export=ALL
#SBATCH --array=0-3
#SBATCH -o ./log/NDE_GAMA_MOCK_15_60_50_0.0003_40_20_12_0_3.%a.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_mock.py --multijobs=True --seed_low=0 --seed_high=3 --n_samples=10000 --num_transforms=15 --num_bins=50 --hidden_features=60 --output_dir=./NDE/GAMA/anneal/mock2/lr3e-4_ann12_zscore_40e_15t_60f/ --add_noise=True --max_lr=0.0003 --max_epochs=40 --anneal_coeff=20 --anneal_tau=12


now=$(date +"%T")
echo "end time ... $now"
