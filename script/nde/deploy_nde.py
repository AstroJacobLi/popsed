'''
python script to deploy slurm jobs for training ndes
'''
import os
import sys
import fire


def deploy_training_job(seed_low, seed_high, name='NMF', output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise/'):
    ''' create slurm script and then submit 
    '''
    time = "24:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J NDE_%s_{seed_low}_{seed_high}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=16G",
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH -o NDE_%s_{seed_low}_{seed_high}.o" % (name),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=jiaxuanl@princeton.edu",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "module purge",
        ". /home/jiaxuanl/Research/popsed/script/setup_env.sh",
        "",
        f"python train_nde.py --seed_low={seed_low} --seed_high={seed_high} --output_dir={output_dir}",
        "",
        "",
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


# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_new/' --seed_low=0 --seed_high=10
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_new/' --seed_low=11 --seed_high=20

####### Large is 50000 samples ######
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=0 --seed_high=5
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=5 --seed_high=10
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=10 --seed_high=15
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=15 --seed_high=20

# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=20 --seed_high=25
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=25 --seed_high=30
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=30 --seed_high=35
# python deploy_nde.py --output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/' --seed_low=35 --seed_high=40
