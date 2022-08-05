#!/bin/bash
#SBATCH -J NMF_ZH_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH -o NMF_ZH_test.o
#SBATCH --mail-type=all
#SBATCH --mail-user=jiaxuanl@princeton.edu

now=$(date +"%T")
echo "start time ... $now"

module purge
. /home/jiaxuanl/Research/popsed/script/setup_env.sh

python gen_spec_nmf_zh.py --version='0.2' --name=NMF_ZH --ncpu=32 --ibatch=test --N_samples=100000 --burst=True --dat_dir='/scratch/gpfs/jiaxuanl/Data/popsed/train_sed_NMF_ZH/nmf_zh_seds'
now=$(date +"%T")
echo "end time ... $now"
