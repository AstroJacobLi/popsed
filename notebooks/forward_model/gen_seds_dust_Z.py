import numpy as np
import pickle
import kuaizi
kuaizi.set_matplotlib(style='nature', usetex=False)
kuaizi.set_env(project='popsed', name='', data_dir='/scratch/gpfs/jiaxuanl/Data')


from prospect.sources.constants import cosmo #WMAP9
import sys
sys.path.append('/home/jiaxuanl/Research/popsed/')

from popsed import mock

import time

# SDSS filters
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']
import itertools

dlambda_spec = 2
wave_lo = 1500
wave_hi = 12000
wavelengths = np.arange(wave_lo, wave_hi, dlambda_spec)

zred_set = np.array([0.])
mass_set = np.array([1]) # Fix mass to 1 M_\odot #10**(np.linspace(9, 12, 20))
tage_set = np.linspace(1, float(cosmo.age(zred_set[0]).value), 30)
tau_set = 10**(np.arange(-2, 2, 0.1))
log_Z_set = np.linspace(-2, 0.5, 30)
dust2_set = np.linspace(0, 4, 30)

print('Total number of samples in parameter space:')
print(len(tau_set) * len(tage_set) * len(mass_set) * len(zred_set) * len(log_Z_set) * len(dust2_set))


sps = mock.build_sps(add_realism=False)

start = time.time()

obs_set = []
# we start with a simple model: tau-SFH, no dust attenuation and emission, no nebular emission
for zred, mass, tage, tau, logzsol, dust2 in itertools.product(zred_set, mass_set, tage_set, 
                                                               tau_set, log_Z_set, dust2_set):
    model = mock.build_model(mass=mass, zred=zred, logzsol=logzsol,
                             sfh=1, tage=tage, tau=tau, 
                             add_dustabs=True, dust2=dust2,
                             uniform_priors=True)
    obs = mock.build_obs(sps, model, add_noise=False, 
                         dlambda_spec=dlambda_spec, wave_lo=wave_lo, wave_hi=wave_hi,
                         # we use true_spectrum and true_mock for emulation
                         # Noise can be added afterward, with a better noise model
                         snr_spec=10, snr_phot=20,
                         filterset=sdss, 
                         continuum_optimize=False)
    obs_set.append(obs)

with open('./train_sed/train_sed_dust_Z_sbatch.pkl', 'wb') as f:
    pickle.dump(obs_set, f)
    f.close()

end = time.time()

print('Time elapsed: {0}'.format(end - start))
