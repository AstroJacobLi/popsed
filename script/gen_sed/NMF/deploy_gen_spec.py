'''
python script to deploy slurm jobs for constructing training spectra
'''
import os
import sys
import fire


def deploy_training_job(ibatch, name='NMF_ZH', python_file='gen_spec_nmf_zh.py', ncpu=32, nsamples=10000, burst=True, output_dir='train_sed_NMF_ZH'):
    ''' create slurm script and then submit 
    '''
    time = "24:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J %s_{ibatch}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=%i" % ncpu,
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH -o %s_{ibatch}.o" % (name),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=jiaxuanl@princeton.edu",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "module purge",
        ". /home/jiaxuanl/Research/popsed/script/setup_env.sh",
        "",
        f"python {python_file} --version='0.2' --name={name} --ncpu={ncpu} --ibatch={ibatch} --N_samples={nsamples} --burst={burst} --dat_dir='/scratch/gpfs/jiaxuanl/Data/popsed/{output_dir}/nmf_zh_seds'",
        ""
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    # create the slurm script execute it and remove it
    f = open('_train.slurm', 'w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    #os.system('rm _train.slurm')
    return None


if __name__ == '__main__':
    fire.Fire(deploy_training_job)


# python deploy_gen_spec.py --name='NMF' --ibatch=1 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=2 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=3 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=4 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=5 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=6 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=7 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=8 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=9 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=10 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch='test' --ncpu=32 --nsamples=100000 --burst=True


### 08.01.2022: generate specs for NMF_ZH
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=1 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=2 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=3 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=4 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=5 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=6 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=7 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=8 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=9 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch=10 --ncpu=32 --nsamples=300000 --burst=True
# python deploy_gen_spec.py --name='NMF_ZH' --ibatch='test' --ncpu=32 --nsamples=100000 --burst=True
