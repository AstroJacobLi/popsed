'''
python script to deploy slurm jobs for constructing training spectra
'''
import os
import sys
import fire


def deploy_training_job(ibatch, name='NMF', ncpu=32, nsamples=10000, burst=True):
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
        f"python gen_spec_nmf.py --name={name} --ncpu={ncpu} --ibatch={ibatch} --N_samples={nsamples} --burst={burst}",
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


# python deploy_gen_spec.py --name='NMF.noburst' --ibatch=test --ncpu=1 --nsamples=1

# python deploy_gen_spec.py --name='NMF' --ibatch=1 --ncpu=32 --nsamples=10000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=11 --ncpu=32 --nsamples=200000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=12 --ncpu=32 --nsamples=200000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=13 --ncpu=32 --nsamples=200000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=14 --ncpu=32 --nsamples=200000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=15 --ncpu=32 --nsamples=200000 --burst=True

# python deploy_gen_spec.py --name='NMF' --ibatch=16 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=17 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=18 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=19 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=20 --ncpu=32 --nsamples=100000 --burst=True

# python deploy_gen_spec.py --name='NMF' --ibatch=21 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=22 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=23 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=24 --ncpu=32 --nsamples=100000 --burst=True
# python deploy_gen_spec.py --name='NMF' --ibatch=25 --ncpu=32 --nsamples=100000 --burst=True

# python deploy_gen_spec.py --name='NMF.noburst' --ibatch=2 --ncpu=32 --nsamples=10000
# python deploy_gen_spec.py --name='NMF.noburst' --ibatch=3 --ncpu=32 --nsamples=10000
