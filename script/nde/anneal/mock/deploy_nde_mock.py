'''
python script to deploy slurm jobs for training ndes

Using CDF transform

Don't use NSA redshift
'''
import os
import sys
import fire


def deploy_training_job(seed_low, seed_high, multijobs=True, python_file='train_nde_mock.py',
                        name='GAMA_MOCK', n_samples=10000, num_bins=40, num_transforms=20, hidden_features=100,
                        add_noise=True, max_lr=3e-4, max_epochs=30, anneal_coeff=20, anneal_tau=10,
                        output_dir='./NDE/NMF/nde_theta_NMF_NSA_freez/'):
    ''' create slurm script and then submit 
    '''
    time = "3:00:00"
    name = '_'.join([str(item) for item in [
                    name, num_transforms, hidden_features, num_bins, max_lr, max_epochs, anneal_coeff, anneal_tau]])

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J NDE_%s_{seed_low}_{seed_high}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=12G",
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

# 22.09.14

# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30/' --seed_low=2 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5  --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr5e-4_ann7p5_30/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=5e-4 --anneal_tau=7.5  --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann6_30/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=6  --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_40/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --max_epochs=40


# The lesson I learned: 10.03.22
# (below all have 50 bins)
# `lr3e-4_ann7p5_30_30t` is bad
# `lr3e-4_ann7p5_30` is better
# `lr3e-4_ann7p5_40` is not better than 30 epoches
# `lr3e-4_ann6_30` is not better than ann7p5


# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30_20b/' --seed_low=0 --seed_high=5 --num_bins=20 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr5e-4_ann7p5_30_20b/' --seed_low=0 --seed_high=5 --num_bins=20 --num_transforms=20 --hidden_features=100 --max_lr=5e-4 --anneal_tau=7.5 --max_epochs=30

# `lr3e-4_ann7p5_30_20b` is bad. 20 bins is bad


# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30_new/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5  --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr2e-4_ann7p5_30_new/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=2e-4 --anneal_tau=7.5  --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann9_30_new/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=9  --max_epochs=30


# Make dust2 and dust_index correlated with each other: lr3e-4_ann7p5_30_corrdust is the best
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30_corrdust/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_35_corrdust/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --anneal_coeff=35 --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_15_corrdust/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --anneal_coeff=35 --max_epochs=30


# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30_corrdust_bound/' --seed_low=0 --seed_high=5 --num_bins=50 --num_transforms=20 --hidden_features=100 --max_lr=3e-4 --anneal_tau=7.5 --anneal_coeff=20 --max_epochs=30
# python deploy_nde_mock.py --output_dir='./NDE/GAMA/anneal/mock/lr3e-4_ann7p5_30_corrdust_bound_100_20_5-/' --seed_low=0 --seed_high=5 --num_bins=100 --num_transforms=20 --hidden_features=50 --max_lr=3e-4 --anneal_tau=7.5 --anneal_coeff=20 --max_epochs=30
