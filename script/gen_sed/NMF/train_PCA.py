import numpy as np
import pickle
import matplotlib.pyplot as plt
import astropy.units as u
import os
os.chdir('/scratch/gpfs/jiaxuanl/Data/popsed/')

import sys
sys.path.append('/home/jiaxuanl/Research/popsed/')
from popsed import mock
from popsed.speculator import SpectrumPCA
from scipy.stats import norm

wave = np.load('./train_sed_NMF/nmf_seds/fsps.wavelength.npy')

for i_bin in range(0, 1):
    print('i_bin =', i_bin)
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
    n_comp = [80, 50, 50, 50, 50][i_bin]

    pca = SpectrumPCA(n_comp,
                      [f'./train_sed_NMF/nmf_seds/fsps.NMF.v0.2.log10spectrum.seed{k+1}{str_wbin}.npy' for k in range(
                          0, 3)]
                      )
    pca.scale_spectra()
    pca.train_pca(chunk_size=1000)
    del pca.normalized_logspec
    pca.save(f'./train_sed_NMF/nmf_seds/fsps.NMF.pca_trained{str_wbin}.pkl')
