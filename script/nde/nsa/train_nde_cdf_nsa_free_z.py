"""
Using CDF transform

Don't provide NSA redshift
"""
import os
import sys
import pickle
import corner
import numpy as np
from tqdm import trange
import fire
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split

os.chdir('/scratch/gpfs/jiaxuanl/Data/popsed/')
sys.path.append('/home/jiaxuanl/Research/popsed/')
from popsed.speculator import SuperSpeculator
import popsed
popsed.set_matplotlib(style='JL', usetex=False, dpi=80)
from popsed import prior

wave = np.load('./train_sed_NMF/nmf_seds/fsps.wavelength.npy')
speculator = SuperSpeculator(
    speculators_dir=[
        f'./train_sed_NMF/best_emu/speculator_best_recon_model_NMF.emu_{i_bin}.pkl' for i_bin in range(0, 5)],
    str_wbin=[
        '.w1000_2000',
        '.w2000_3600',
        '.w3600_5500',
        '.w5500_7410',
        '.w7410_60000'],
    wavelength=wave,
    params_name=['kappa1_sfh', 'kappa2_sfh', 'kappa3_sfh',
                 'fburst', 'tburst', 'logzsol', 'dust1', 'dust2',
                 'dust_index', 'redshift', 'logm'])

noise = 'nsa'
noise_model_dir = './noise_model/nsa_noise_model_mag_chang.npy'
filters = ['sdss_{0}0'.format(b) for b in 'ugriz']

# Load NSA data
mags_nsa = np.load('./nsa_sedflow/nsa_mag_data.npy').T
flag = ((mags_nsa[:, 0] - mags_nsa[:, 1]) > 0)
flag &= ((mags_nsa[:, 1] - mags_nsa[:, 2]) > 0)
flag &= ((mags_nsa[:, 1] - mags_nsa[:, 4]) > 0)
flag &= ((mags_nsa[:, 2] - mags_nsa[:, 3]) > 0)
flag &= ((mags_nsa[:, 2] - mags_nsa[:, 4]) > 0)
mags_nsa = mags_nsa[flag]

# z_nsa = np.load('./nsa_sedflow/nsa_redshift.npy')[flag]
# sedflow_sample = np.load('./nsa_sedflow/nsa_sedflow.posterior.reorganized.npy')
# z_nsa = np.random.choice(sedflow_sample[:, -2], len(mags_nsa))

X_data = torch.Tensor(mags_nsa[:, :])
print('Total number of samples:', len(X_data))

# Determine the intrinsic sampling loss
X_datas = []
for i in range(2):
    ind = np.random.randint(0, len(X_data), 5000)
    X_datas.append(X_data[ind].to('cuda'))

from geomloss import SamplesLoss
L = SamplesLoss(loss='sinkhorn', **{'p': 1, 'blur': 0.1, 'scaling': 0.5})
intr_loss = L(X_datas[0], X_datas[1]).item()
print("Intrinsic sampling loss:", intr_loss)


_prior_NDE = speculator.bounds.copy()
_prior_NDE[-2] = np.array([0.01, 0.055])
_prior_NDE[-1] = np.array([7.5, 12.5])

# _prior_NDE = np.vstack([_prior_NDE[:-2], _prior_NDE[-1:]])


def train_NDEs(seed_low, seed_high, num_transforms=5, num_bins=40, hidden_features=100,
               only_penalty=False, output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_large/'):
    # Start train NDEs
    from popsed.nde import WassersteinNeuralDensityEstimator

    _bounds = np.zeros_like(speculator.bounds)
    _bounds = np.zeros_like(_bounds)
    _bounds = np.vstack([-np.abs(np.random.normal(size=len(_bounds)) / 30),
                         np.abs(np.random.normal(size=len(_bounds)) / 30)]).T
    _stds = np.ones(len(_bounds))

    for seed in range(seed_low, seed_high):
        X_train, X_vali = train_test_split(X_data, test_size=0.05)
        Y_train = torch.ones(len(X_train), 11)

        NDE_theta = WassersteinNeuralDensityEstimator(method='nsf',
                                                      name='NMF',
                                                      num_transforms=num_transforms,  # 15,  # 10
                                                      num_bins=num_bins,  # 10,  # how smashed it is. 10
                                                      hidden_features=hidden_features,  # 120,
                                                      seed=seed,
                                                      output_dir=output_dir,
                                                      initial_pos={'bounds': _bounds,
                                                                   'std': _stds,
                                                                   },
                                                      normalize=False,
                                                      regularize=True,
                                                      NDE_prior=_prior_NDE)
        NDE_theta.build(
            Y_train,
            X_train,
            filterset=filters,
            optimizer='adam')
        NDE_theta.load_validation_data(X_vali)
        NDE_theta.bounds = speculator.bounds
        NDE_theta.params_name = speculator.params_name

        NDE_theta.external_redshift_data = None  # z_nsa

        print('Total number of params in the model:',
              sum(p.numel() for p in NDE_theta.net.parameters() if p.requires_grad))

        max_epochs = 5
        # blurs = [0.2, 0.1, 0.1, 0.1, 0.1]
        blurs = [0.2, 0.1, 0.1, 0.05, 0.05]

        try:
            print('### Training NDE for seed {0}'.format(seed))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(NDE_theta.optimizer,
                                                            max_lr=8e-4,
                                                            steps_per_epoch=100,
                                                            epochs=max_epochs)
            for i, epoch in enumerate(range(max_epochs)):
                print('    Epoch {0}'.format(epoch))
                print('    lr:', NDE_theta.optimizer.param_groups[0]['lr'])
                NDE_theta.train(n_epochs=100,
                                speculator=speculator,
                                only_penalty=only_penalty,
                                noise=noise, noise_model_dir=noise_model_dir,
                                sinkhorn_kwargs={
                                    'p': 1, 'blur': blurs[i], 'scaling': 0.5},
                                scheduler=scheduler
                                )
            print(f'    Succeeded in training for {max_epochs} epochs!')
            print('    Saving NDE model for seed {0}'.format(seed))
            print('\n\n')
            NDE_theta.save_model(
                os.path.join(NDE_theta.output_dir,
                             f'nde_theta_last_model_{NDE_theta.method}_{NDE_theta.seed}.pkl')
            )
        except Exception as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(train_NDEs)
