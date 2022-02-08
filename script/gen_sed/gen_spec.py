import numpy as np
import pickle
import matplotlib.pyplot as plt
import astropy.units as u
import kuaizi
import fire

from sedpy.observate import load_filters
from prospect.sources.constants import cosmo #WMAP9
import os, sys
import multiprocess as mp
from functools import partial
import time

mp.freeze_support()

kuaizi.set_env(project='popsed', name='', data_dir='/scratch/gpfs/jiaxuanl/Data')
sys.path.append('/home/jiaxuanl/Research/popsed/')

from popsed import mock
from popsed.sfh import params_to_sfh, parametric_sfr, parametric_mwa

dlambda_spec = 2
wave_lo = 1000
wave_hi = 15000
wavelengths = np.arange(wave_lo, wave_hi, dlambda_spec)


def _fsps_model_wrapper(theta, sps):
    tage, tau, logzsol, dust2, mass, zred = theta
    model = mock.build_model(mass=mass, zred=zred, logzsol=logzsol,
                             sfh=1, tage=tage, tau=tau, 
                             add_dustabs=True, dust2=dust2,
                             uniform_priors=True)
    obs = mock.build_obs(sps, model, 
                         add_noise=False, 
                         dlambda_spec=dlambda_spec, wave_lo=wave_lo, wave_hi=wave_hi,
                         # we use true_spectrum and true_mock for emulation
                         # Noise can be added afterward
                         snr_spec=10, snr_phot=20,
                         filterset=None,
                         continuum_optimize=False)
    return np.log10(obs['true_spectrum']) 
    

def gen_spec(ncpu=32, ibatch=1, N_samples=5000, 
             name='TDZ', version='0.1',  # tau-z-dust
             dat_dir='/scratch/gpfs/jiaxuanl/Data/popsed/train_sed/',):
    start = time.time()

    ftheta = os.path.join(dat_dir, 'fsps.%s.v%s.theta.seed%s.npy' % (name, version, ibatch)) 
    fspectrum = os.path.join(dat_dir, 'fsps.%s.v%s.log10spectrum.seed%s.npy' % (name, version, ibatch)) 
    print('Initialize SPS')
    sps = mock.build_sps(add_realism=False)

    if ibatch == 'test':
        np.random.seed(1234)
    else:
        np.random.seed(ibatch)
    zred_set = np.random.uniform(0.0, 0.5, N_samples)
    mass_set = np.random.uniform(1.0, 1.0, N_samples) # Fix mass to 1 M_\odot
    tage_set = np.random.uniform(0, 1, N_samples) * cosmo.age(zred_set).value
    tau_set = 10**(np.random.uniform(-2, 2, size=N_samples))
    log_Z_set = np.random.uniform(-2, 0.5, size=N_samples)
    dust2_set = np.random.uniform(0, 3, size=N_samples)

    print('Total number of samples in parameter space:')
    print(len(tau_set))
    thetas = np.vstack([tage_set, tau_set, log_Z_set, dust2_set, mass_set, np.zeros_like(zred_set)]).T

    print()  
    print('--- batch %s ---' % str(ibatch)) 
    # save parameters sampled from prior 
    print('  saving thetas to %s' % ftheta)
    np.save(ftheta, thetas)

    if (ncpu == 1): # run on serial 
        logspectra = []
        for _theta in thetas:
            logspectra.append(_fsps_model_wrapper(_theta, sps))
    else: 
        pewl = mp.Pool(ncpu) 
        logspectra = pewl.map(partial(_fsps_model_wrapper, sps=sps), thetas) 

    print('  saving log10(spectra) to %s' % fspectrum)
    np.save(fspectrum, np.array(logspectra))
    print()  

    end = time.time()
    print('Elapsed time = ', end - start)

if __name__ == '__main__':
    fire.Fire(gen_spec)