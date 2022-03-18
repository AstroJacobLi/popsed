"""
Script for training the spectra emulator. 
The notebook for this script is `popsed/notebooks/forward_model/NMF/spec_phot_emulator.ipynb`.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import fire
os.chdir('/scratch/gpfs/jiaxuanl/Data/popsed/')
import torch

import sys
sys.path.append('/home/jiaxuanl/Research/popsed/')
from popsed.speculator import Speculator
from scipy.stats import norm


def _train_emu(i_bin, name='NMF', file_low=0, file_high=15, batch_size=256, rounds=6):
    training_sed_version = '0.2'
    
    if i_bin == 0:
        raise ValueError("For your purpose, you don't need to train the very blue end of the spectrum.")
    
    # Load params and specs
    print(f'Loading from file {file_low} to {file_high}')
    params = np.concatenate([np.load(f'./train_sed_NMF/fsps.NMF.v{training_sed_version}.theta_unt.seed{i+1}.npy')
                            for i in range(int(file_low), int(file_high))])
    # exclude stellar mass (1 M_\dot), remain redshift (i.e., length of SFH)
    params = params[:, 1:]
    # In total, 10 parameters as input of the emulator
    print('Number of specs=', len(params))
    print('Parameter dimension=', params.shape)
    wave = np.load('./train_sed_NMF/fsps.wavelength.npy')
    wave_bin = [
        (wave >= 1000) & (wave < 2000),
        (wave >= 2000) & (wave < 3600),
        (wave >= 3600) & (wave < 5500),
        (wave >= 5500) & (wave < 7410),
        (wave >= 7410) & (wave < 60000)
    ][i_bin]
    str_wbin = [
        '.w1000_2000',
        '.w2000_3600',
        '.w3600_5500',
        '.w5500_7410',
        '.w7410_60000'
    ][i_bin]

    # Initialize the speculator object, load PCA
    speculator = Speculator(name=f'{name}_{i_bin}', model='NMF', n_parameters=10,
                            pca_filename=f'./train_sed_NMF/fsps.NMF.pca_trained{str_wbin}.pkl',
                            hidden_size=[256, 256, 256, 256])

    # Load training spectra in a clever way. Because we train 
    # the emulator to predict PCA coefficients, we readin 
    # spectra but only keep the corresponding PCA coefficients.
    # Otherwise the memory will overflow.
    fspecs = [f'./train_sed_NMF/fsps.NMF.v{training_sed_version}.log10spectrum.seed{i+1}{str_wbin}.npy'
                           for i in range(int(file_low), int(file_high))]
    pca_coeff = []
    for file in fspecs:
        pca_coeff.append(speculator.pca.PCA.transform(speculator.pca.logspec_scaler.transform(np.load(file, mmap_mode='r'))))
    pca_coeff = np.concatenate(pca_coeff)
    print('PCA coeff shape=', pca_coeff.shape)

    speculator.load_data(pca_coeff, params,
                         params_name=['beta1_sfh', 'beta2_sfh', 'beta3_sfh', 'beta4_sfh',
                                      'fburst', 'tburst', 'logzsol',
                                      'dust1', 'dust2', 'dust_index', 'redshift'],
                         val_frac=0.1, batch_size=batch_size,
                         wave_rest=torch.Tensor(wave[wave_bin]),
                         wave_obs=torch.Tensor(wave[wave_bin]))
    train_ind = speculator.dataloaders['train'].dataset.indices
    val_ind = speculator.dataloaders['val'].dataset.indices

    # Training
    print('Now training emulator with batch_size =', batch_size)
    lrs = np.array([1e-3, 5e-4, 1e-4, 5e-5, 1e-5])[:int(rounds)]
    n_ep = [100 for _ in lrs]

    i = 0
    for n, lr in zip(n_ep, lrs):
        i += 1
        print(f'\n \n #### Round {i}, Learning rate = {lr} ####')
        speculator.train(learning_rate=lr, n_epochs=n)

    # Plot and save the loss curve
    speculator.plot_loss()
    plt.savefig(f'./{name}_emulator_{str_wbin}_loss.png')
    plt.close()

    # Validation
    _specs = speculator.predict_spec(torch.Tensor(
        params[val_ind]).to('cuda')).cpu().detach().numpy()
    
    val_logspec = speculator.pca.logspec_scaler.inverse_transform(speculator.pca.PCA.inverse_transform(pca_coeff[val_ind]))
    diff = (10**val_logspec - _specs) / 10**val_logspec * 100

    x = wave[wave_bin] / 10
    plt.plot(x, np.nanmedian(diff, axis=0), color='r', label='Median')
    plt.fill_between(x,
                     np.nanpercentile(diff, (1 - norm.cdf(1)) * 100, axis=0),
                     np.nanpercentile(diff, norm.cdf(1) * 100, axis=0), alpha=0.5, color='tomato', label=r'$1\sigma$')

    plt.fill_between(x,
                     np.nanpercentile(diff, (1 - norm.cdf(2)) * 100, axis=0),
                     np.nanpercentile(diff, norm.cdf(2) * 100, axis=0), alpha=0.2, color='salmon', label=r'$2\sigma$')
    plt.ylabel(
        r'% fractional error $(l_{\lambda}^{\mathrm{NN}} - l_{\lambda})/l_{\lambda}$')
    plt.xlabel('Wavelength [nm]')
    plt.legend()
    plt.savefig(f'./{name}_emulator_{str_wbin}_frac_err.png')
    plt.close()


if __name__ == '__main__':
    fire.Fire(_train_emu)
