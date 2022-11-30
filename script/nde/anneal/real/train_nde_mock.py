"""
Using CDF transform

Use GAMA DR3 aperture matched photometry
"""
import os
import sys
import numpy as np
from tqdm import trange
import fire

import torch
from sklearn.model_selection import train_test_split

os.chdir('/scratch/gpfs/jiaxuanl/Data/popsed/')
sys.path.append('/home/jiaxuanl/Research/popsed/')
from popsed.speculator import SuperSpeculator, StandardScaler
from torch.utils.data import DataLoader
from geomloss import SamplesLoss
import gc

ismock = True
PHOT = 'AUTO'
name = 'NMF'
wave = np.load(f'./train_sed_{name}/{name.lower()}_seds/fsps.wavelength.npy')

if name == 'NMF_ZH':
    spec_dir = [
        f'./train_sed_{name}/best_emu/speculator_best_recon_model_{name}_{i_bin}.pkl' for i_bin in range(0, 5)]
    params_name = ['kappa1_sfh', 'kappa2_sfh', 'kappa3_sfh',
                   'fburst', 'tburst', 'gamma1_zh', 'gamma2_zh',
                   'dust1', 'dust2',
                   'dust_index', 'redshift', 'logm']
else:
    spec_dir = [
        f'./train_sed_{name}/best_emu/speculator_best_recon_model_{name}.emu_{i_bin}.pkl' for i_bin in range(0, 5)]
    params_name = ['kappa1_sfh', 'kappa2_sfh', 'kappa3_sfh',
                   'fburst', 'tburst', 'logzsol',
                   'dust1', 'dust2',
                   'dust_index', 'redshift', 'logm']
speculator = SuperSpeculator(
    speculators_dir=spec_dir,
    str_wbin=['.w1000_2000',
              '.w2000_3600',
              '.w3600_5500',
              '.w5500_7410',
              '.w7410_60000'],
    wavelength=wave,
    params_name=params_name,
    device='cuda', use_speclite=True)
gama_filters = ['sdss2010-{0}'.format(b) for b in 'ugriz']
speculator._calc_transmission(gama_filters)

# Load NSA data
if ismock:
    X_data = np.load('./NDE/GAMA/NMF/mock_mags_dr3apmatch_noise_2.npy')[:, :5]
else:
    X_data = np.load(
        f'./reference_catalog/GAMA/gama_clean_mag_dr3_apmatch_{PHOT}_snr1.npy')[:, :5]
print('Photometry used:', PHOT)
print('Total number of samples:', len(X_data))
gc.collect()
torch.cuda.empty_cache()

# Determine the intrinsic sampling loss
scaler = StandardScaler(device='cpu')
scaler.fit(X_data)
_X_data = scaler.transform(X_data)
# Determine the intrinsic sampling loss
X_datas = []
for i in range(5):
    ind = np.random.randint(0, len(_X_data), 10000)
    X_datas.append(torch.Tensor(_X_data[ind]).to('cuda'))
L = SamplesLoss(loss='sinkhorn', **{'p': 1, 'blur': 0.002, 'scaling': 0.9})
intr_loss = []
for i in range(5):
    dataloader = DataLoader(_X_data, batch_size=10000, shuffle=True)
    data_loss = 0.
    for x in dataloader:
        data_loss += L(X_datas[i], x.to('cuda'))
    loss = data_loss / len(dataloader)
    intr_loss.append(loss.item())
print("Intrinsic sampling loss:", np.mean(intr_loss), '+-', np.std(intr_loss))
del X_datas
gc.collect()
torch.cuda.empty_cache()

_prior_NDE = speculator.bounds.copy()
_prior_NDE[-2] = np.array([0., 0.8])
_prior_NDE[-1] = np.array([7.5, 13.0])


def train_NDEs(seed_low, seed_high, multijobs=False, n_samples=5000, num_transforms=20,
               num_bins=40, hidden_features=100, add_noise=True, max_lr=3e-4, max_epochs=30,
               anneal_coeff=20, anneal_tau=7.5,
               add_penalty=False, output_dir='./NDE/GAMA/{name}/nde_theta_{name}_real/'):
    # Start train NDEs
    from popsed.nde import WassersteinNeuralDensityEstimator

    if add_noise:
        noise = 'gama_snr'
    else:
        noise = None

    noise_model_dir = f'./noise_model/gama_snr_model_mag_dr3_apmatch_{PHOT}.npy'

    if multijobs == False:
        seed_range = trange(seed_low, seed_high)
    else:
        seed_range = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
        print('Seed range:', seed_range)
    for seed in seed_range:
        _bounds = np.zeros_like(speculator.bounds)
        _bounds = np.zeros_like(_bounds)
        _bounds = np.vstack([-np.abs(np.random.normal(size=len(_bounds)) / 10),
                            np.abs(np.random.normal(size=len(_bounds)) / 10)]).T
        _stds = np.ones(len(_bounds))

        X_train, X_vali = train_test_split(X_data, test_size=0.05)
        if name == 'NMF_ZH':
            Y_train = torch.ones(len(X_train), 12)
        else:
            Y_train = torch.ones(len(X_train), 11)

        NDE_theta = WassersteinNeuralDensityEstimator(method='nsf',
                                                      name=name,
                                                      num_transforms=num_transforms,
                                                      num_bins=num_bins,
                                                      hidden_features=hidden_features,
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
            z_score=True,
            filterset=gama_filters,
            optimizer='adam')
        NDE_theta.load_validation_data(X_vali)
        NDE_theta.bounds = speculator.bounds
        NDE_theta.params_name = speculator.params_name
        NDE_theta.external_redshift_data = None  # z_nsa

        print('Total number of params in the model:',
              sum(p.numel() for p in NDE_theta.net.parameters() if p.requires_grad))

        max_epochs = max_epochs
        blurs = [0.3, 0.3, 0.2, 0.2, 0.1, 0.1,
                 0.1, 0.05, 0.05, 0.05] + [0.002] * max_epochs
        snrs = [1 + anneal_coeff * np.exp(- anneal_tau / max_epochs * i)
                for i in range(max_epochs)]  # larger anneal_coeff, after annealing
        steps = 30

        try:
            print('### Training NDE for seed {0}'.format(seed))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(NDE_theta.optimizer,
                                                            max_lr=max_lr,
                                                            steps_per_epoch=steps,
                                                            epochs=max_epochs,
                                                            div_factor=10, final_div_factor=100)
            for i, epoch in enumerate(range(max_epochs)):
                np.save(os.path.join(NDE_theta.output_dir, f'{NDE_theta.method}_{NDE_theta.seed}_sample_{i+1}.npy'),
                        NDE_theta.sample(5000).detach().cpu().numpy())

                print('    Epoch {0}'.format(epoch))
                print('\n\n')
                print(
                    '    lr:', NDE_theta.optimizer.param_groups[0]['lr'], '  blurs:', blurs[i], ' snrs:', snrs[i])
                NDE_theta.train(n_epochs=steps,
                                speculator=speculator,
                                add_penalty=add_penalty,
                                n_samples=n_samples,
                                noise=noise,
                                noise_model_dir=noise_model_dir,
                                SNR=snrs[i],
                                sinkhorn_kwargs={
                                    'p': 1, 'blur': blurs[i], 'scaling': 0.9},
                                scheduler=scheduler
                                )
                if NDE_theta.vali_loss_history[-1] > 20:
                    raise ValueError(
                        'Validation loss is too large! Next seed!')
                NDE_theta.save_model(
                    os.path.join(NDE_theta.output_dir,
                                 f'nde_theta_last_model_{NDE_theta.method}_{NDE_theta.seed}.pkl')
                )
            print(f'    Succeeded in training for {max_epochs} epochs!')
            print('    Saving NDE model for seed {0}'.format(seed))
            print('\n\n')
            np.save(os.path.join(NDE_theta.output_dir, f'{NDE_theta.method}_{NDE_theta.seed}_sample_{i+1}.npy'),
                    NDE_theta.sample(5000).detach().cpu().numpy())

        except Exception as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(train_NDEs)

# python train_nde_mock.py --seed_low=0 --seed_high=1 --n_samples=10000 --num_transforms=20 --num_bins=40 --hidden_features=100 --max_lr=3e-4 --output_dir=./NDE/GAMA/anneal/mock/lr3e-4_exp0p25/
# python train_nde_mock.py --seed_low=0 --seed_high=1 --n_samples=10000 --num_transforms=20 --num_bins=40 --hidden_features=100 --max_lr=3e-4 --output_dir=./NDE/GAMA/anneal/mock/lr3e-4_exp0p25/
