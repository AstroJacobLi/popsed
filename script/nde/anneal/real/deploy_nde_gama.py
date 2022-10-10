'''
python script to deploy slurm jobs for training ndes

Using CDF transform

Don't use NSA redshift
'''
import os
import sys
import fire


def deploy_training_job(seed_low, seed_high, multijobs=True, python_file='train_nde_gama.py',
                        name='GAMA_REAL', n_samples=10000, num_bins=40, num_transforms=20, hidden_features=100,
                        add_noise=True, max_lr=3e-4, max_epochs=30, anneal_coeff=20, anneal_tau=10,
                        output_dir='./NDE/NMF/nde_theta_NMF_NSA_freez/'):
    ''' create slurm script and then submit 
    '''
    time = "1:30:00"
    name = '_'.join([str(item) for item in [
                    name, num_transforms, hidden_features, num_bins, max_lr, max_epochs, anneal_coeff, anneal_tau]])

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J NDE_%s_{seed_low}_{seed_high}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=8G",
        "#SBATCH --time=%s" % time,
        "#SBATCH --export=ALL",
        f"#SBATCH --array={seed_low}-{seed_high}" if multijobs else "",
        f"#SBATCH -o ./log/NDE_{name}_{seed_low}_{seed_high}.%a.o",
        # "#SBATCH --output=slurm-%A.%a.out",
        "#SBATCH --mail-type=all",
        "#SBATCH --mail-user=jiaxuanl@princeton.edu",
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        'echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."',
        'echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"',
        'echo "Executing on the machine:" $(hostname)',
        "",
        "module purge",
        ". /home/jiaxuanl/Research/popsed/script/setup_env.sh",
        "",
        f"python {python_file} --multijobs={multijobs} --seed_low={seed_low} --seed_high={seed_high} --n_samples={n_samples} --num_transforms={num_transforms} --num_bins={num_bins} --hidden_features={hidden_features} --output_dir={output_dir} --add_noise={add_noise} --max_lr={max_lr} --max_epochs={max_epochs} --anneal_coeff={anneal_coeff} --anneal_tau={anneal_tau}",
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


# python deploy_nde_gama.py --output_dir='./NDE/GAMA/anneal/real/lr3e-4_ann12_zscore_40e/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=12 --max_epochs=40
