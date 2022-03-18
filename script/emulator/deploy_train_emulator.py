'''
python script to deploy slurm jobs for training emulator
'''
import os
import sys
import fire


def deploy_training_job(ibatch, name='NMF.emu', batch_size=256, file_low=0, file_high=15, rounds=6):
    ''' create slurm script and then submit 
    '''
    time = "24:00:00"

    cntnt = '\n'.join([
        "#!/bin/bash",
        f"#SBATCH -J %s_{ibatch}" % (name),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=48G",
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
        f"python train_emulator.py --name={name} --i_bin={ibatch} --file_low={file_low} --file_high={file_high} --batch_size={batch_size} --rounds={rounds}",
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
# arch = 4 x 246
# loss = mse based on logspec

# Converge to recon_err ~ 0.2

# Now train using 2340000 SEDs.
# python deploy_train_emulator.py --ibatch=2 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=25 --rounds=6
# python deploy_train_emulator.py --ibatch=1 --name='NMF.emu' --batch_size=512 --file_low=0 --file_high=25 --rounds=6
