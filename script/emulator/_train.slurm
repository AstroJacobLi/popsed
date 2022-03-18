#!/bin/bash
#SBATCH -J NMF.emu_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o NMF.emu_1.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python train_emulator.py --name=NMF.emu --i_bin=1 --file_low=0 --file_high=25 --batch_size=512 --rounds=6
now=$(date +"%T")
echo "end time ... $now"
