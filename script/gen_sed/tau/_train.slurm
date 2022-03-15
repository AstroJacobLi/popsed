#!/bin/bash
#SBATCH -J TZD_5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH -o TZD_5.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python gen_spec.py --name=TZD --ncpu=32 --ibatch=5 --N_samples=10000
now=$(date +"%T")
echo "end time ... $now"
