'''
python script to deploy slurm jobs for training ndes

Using CDF transform

Don't use NSA redshift
'''
import os
import sys
import fire


def deploy_training_job(seed_low, seed_high, multijobs=False, python_file='train_nde_mock.py',
                        name='GAMA_MOCK', n_samples=10000, num_bins=50, num_transforms=20, hidden_features=100,
                        add_noise=False, smallblur=True,
                        output_dir='./NDE/NMF/nde_theta_NMF_NSA_freez/'):
    ''' create slurm script and then submit 
    '''
    time = "12:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J NDE_%s_{seed_low}_{seed_high}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=22G",
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH --array={seed_low}-{seed_high}" if multijobs else "",
        f"#SBATCH -o ./log/NDE_%s_{seed_low}_{seed_high}_{num_transforms}_{hidden_features}_{num_bins}.o" % (
            name),
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=jiaxuanl@princeton.edu",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "module purge",
        ". /home/jiaxuanl/Research/popsed/script/setup_env.sh",
        "",
        f"python {python_file} --seed_low={seed_low} --seed_high={seed_high} --n_samples={n_samples} --num_transforms={num_transforms} --num_bins={num_bins} --hidden_features={hidden_features} --output_dir={output_dir} --add_noise={add_noise} --smallblur={smallblur}",
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

# 22.09.14

# python deploy_nde_mock.py --output_dir='./NDE/GAMA/NMF/nde_theta_NMF_CDF_mock_flatdirich_nonoise/' --seed_low=0 --seed_high=20 --num_bins=50 --num_transforms=20 --hidden_features=128
