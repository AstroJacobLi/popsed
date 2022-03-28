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
noise_model_dir = './noise_model/nsa_noise_model_mag.npy'


def gen_truth(N_samples=5000):
    print(f'Generating {N_samples} mock params')
    ncomp = 4
    priors = prior.load_priors([
        # Log stellar mass, in M_sun
        prior.GaussianPrior(10.5, 0.4, label='logm'),
        # flat dirichilet priors for SFH
        prior.TruncatedNormalPrior(0, 1, 0.5, 0.1),
        prior.TruncatedNormalPrior(0, 1, 0.5, 0.1),
        prior.TruncatedNormalPrior(0, 1, 0.5, 0.1),
        #         prior.FlatDirichletPrior(ncomp, label='beta'),
        # uniform priors on the mass fraction of burst
        prior.TruncatedNormalPrior(0, 0.6, 0.2, 0.1, label='fburst'),
        # uniform priors on star-burst lookback time
        prior.TruncatedNormalPrior(1e-2, 13.27, 5, 1.5, label='tburst'),
        # uniform priors on log-metallicity, absolute Z
        prior.TruncatedNormalPrior(-2.6, 0.3, -1, 0.3, label='logzsol'),
        # uniform priors on dust1
        prior.TruncatedNormalPrior(0., 3., 0.6, 0.3, label='dust1'),
        # uniform priors on dust2
        prior.TruncatedNormalPrior(0., 3., 0.6, 0.3, label='dust2'),
        # uniform priors on dust_index
        prior.TruncatedNormalPrior(-3., 1., -1, 0.3, label='dust_index'),
        # uniformly sample redshift
        prior.TruncatedNormalPrior(0., 1.5, 0.08, 0.05, label='redshift')
    ])

    _thetas_unt = np.array([priors.sample() for i in range(N_samples)])
    _thetas = np.hstack([_thetas_unt[:, 0:1],
                         prior.FlatDirichletPrior(
                             4).transform(_thetas_unt[:, 1:4]),
                         _thetas_unt[:, 4:]])

    return _thetas, _thetas_unt


# Generate mock obs
_thetas, _thetas_unt = gen_truth(N_samples=7000)
filters = ['sdss_{0}0'.format(b) for b in 'ugriz']
Y_truth = np.hstack([_thetas_unt[:, 1:],  # params taken by emulator, including redshift (for t_age)
                     _thetas_unt[:, 0:1],  # stellar mass
                     ])
Y_truth = torch.Tensor(Y_truth).to('cuda')

X_data = speculator._predict_mag_with_mass_redshift(Y_truth, filterset=filters,
                                                    noise=noise,
                                                    noise_model_dir=noise_model_dir)

flag = ~(torch.isnan(X_data).any(dim=1) | torch.isinf(X_data).any(dim=1))
flags = [((Y_truth[:, i] < speculator.bounds[i, 1]) & (Y_truth[:, i] > speculator.bounds[i, 0])).cpu().numpy()
         for i in range(len(speculator.bounds))]
flag = flag.cpu().numpy() & np.array(flags).all(axis=0)

X_data = X_data[flag].detach()
Y_truth = Y_truth[flag]


def train_NDEs(seed_low, seed_high, output_dir='./NDE/NMF/nde_theta_NMF_sdss_noise_new/'):
    # Start train NDEs
    from popsed.nde import WassersteinNeuralDensityEstimator

    _bounds = speculator.bounds
    for i in range(3):
        _bounds[i] = [0.1, 0.9]
    _bounds[6:8] = [1e-2, 3]
    _bounds[3] = [0.001, 0.999] # fburst
    _bounds[-2] = [1e-3, 0.3] # redshift
    _bounds[-1] = [8, 12] # log_m

    for seed in range(seed_low, seed_high):
        X_train, X_vali, Y_train, Y_vali = train_test_split(
            X_data, Y_truth, test_size=0.1)
        NDE_theta = WassersteinNeuralDensityEstimator(method='nsf',
                                                      name='NMF',
                                                      num_transforms=15,  # 10
                                                      num_bins=15,  # how smashed it is. 10
                                                      hidden_features=50,  # 120,
                                                      seed=seed,
                                                      output_dir=output_dir,
                                                      initial_pos={'bounds': _bounds,
                                                                   'std':  [0.3, 0.3, 0.3, # sfh
                                                                            0.3, 5, # burst 
                                                                            0.5, # logzsol
                                                                            0.6, 0.6, 0.6, # dust
                                                                            0.1, 0.4], # redshift
                                                                   },
                                                      normalize=False)
        NDE_theta.build(
            Y_train,
            X_train,
            filterset=filters,
            optimizer='adam')
        NDE_theta.load_validation_data(X_vali)
        NDE_theta.bounds = speculator.bounds
        NDE_theta.params_name = speculator.params_name

        max_epochs = 6
        scheduler = torch.optim.lr_scheduler.OneCycleLR(NDE_theta.optimizer,
                                                        max_lr=6e-4,
                                                        steps_per_epoch=100,
                                                        epochs=max_epochs)
        try:
            print('### Training NDE for seed {0}'.format(seed))
            for epoch in range(max_epochs):
                print('    Epoch {0}'.format(epoch))
                print('    lr:', NDE_theta.optimizer.param_groups[0]['lr'])
                NDE_theta.train(n_epochs=100,
                                speculator=speculator,
                                only_penalty=False,
                                noise=noise, noise_model_dir=noise_model_dir,
                                sinkhorn_kwargs={
                                    'p': 1, 'blur': 0.1, 'scaling': 0.8},
                                scheduler=scheduler
                                )
            print(f'    Succeeded in training for {max_epochs} epochs! \n\n')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(train_NDEs)
