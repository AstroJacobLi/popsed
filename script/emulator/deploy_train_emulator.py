'''
python script to deploy slurm jobs for training emulator
'''
import os
import sys
import fire


def deploy_training_job(ibatch, name='NMF_ZH.emu', python_file='train_emulator_ZH.py',
                        batch_size=256, file_low=0, file_high=15, rounds=6):
    ''' create slurm script and then submit 
    '''
    time = "24:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J %s_{ibatch}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=56G",
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
        f"python {python_file} --name={name} --i_bin={ibatch} --file_low={file_low} --file_high={file_high} --batch_size={batch_size} --rounds={rounds}",
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


# Now the best parameters are:
# batch_size = 512
# lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5], n_steps in each epoch = 100
# arch = 4 x 256
# loss = mse based on logspec

# Converge to recon_err ~ 0.2

# Now train using 3e6 SEDs. Also train ibatch=0. It will be come important after redshifting.
# For ibatch=0, i use 5 x 256
# For ibatch=1, i use 128 + 4 x 256

# python deploy_train_emulator.py --ibatch=0 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=1 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=2 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=3 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=4 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=10 --rounds=6

# 22.08.03: retrain the emulator using NMF_ZH model
# python deploy_train_emulator.py --ibatch=0 --name='NMF_ZH' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=1 --name='NMF_ZH' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=2 --name='NMF_ZH' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=3 --name='NMF_ZH' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
# python deploy_train_emulator.py --ibatch=4 --name='NMF_ZH' --batch_size=512 --file_low=0 --file_high=10 --rounds=6
