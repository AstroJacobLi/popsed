#!/bin/bash
#SBATCH -J NMF_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o NMF_4.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python gen_spec_nmf.py --name=NMF --ncpu=32 --ibatch=4 --N_samples=10000
now=$(date +"%T")
echo "end time ... $now"
