#!/bin/bash
#SBATCH -J NDE_DR3_mock_20_100_60_0.0003_30_20_8_0_8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=11:59:59
#SBATCH --export=ALL

#SBATCH -o ./log/NDE_DR3_mock_20_100_60_0.0003_30_20_8_0_8.%a.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_nde_mock.py --multijobs=False --seed_low=0 --seed_high=8 --n_samples=10000 --num_transforms=20 --num_bins=60 --hidden_features=100 --output_dir=./NDE/GAMA/anneal/mock/lr3e-4_ann8_30e_20t/ --add_noise=True --max_lr=0.0003 --max_epochs=30 --anneal_coeff=20 --anneal_tau=8


now=$(date +"%T")
echo "end time ... $now"
