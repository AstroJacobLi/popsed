'''
Modified Speculator, based on https://github.com/justinalsing/speculator/blob/master/speculator/speculator.py
'''

import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
from torchinterp1d import Interp1d

from scipy.interpolate import interp1d

from sedpy import observate
# from sklearn.preprocessing import StandardScaler

def flux2mag(flux): 
    ''' convert flux in nanomaggies to magnitudes
    From https://github.com/changhoonhahn/SEDflow/blob/main/src/sedflow/train.py
    '''
    if torch.is_tensor(flux):
        return 22.5 - 2.5 * torch.log10(flux)
    else:
        return 22.5 - 2.5 * np.log10(flux)


def mag2flux(mag): 
    ''' convert magnitudes to flux in nanomaggies
    '''
    return 10**(0.4 * (22.5 - mag)) 


def sigma_flux2mag(sigma_flux, flux): 
    ''' convert sigma_flux to sigma_mag
    '''
    if torch.is_tensor(flux):
        return torch.abs(-2.5 * (sigma_flux) / flux / 2.302585092994046)
    else:
        return np.abs(-2.5 * (sigma_flux) / flux / np.log(10))


def sigma_mag2flux(sigma_mag, mag): 
    ''' convert sigma_mag to sigma_flux
    '''
    flux = mag2flux(mag)
    if torch.is_tensor(mag):
        return torch.abs(flux) * torch.abs(-0.4 * 2.302585092994046 * sigma_mag)
    else:
        return np.abs(flux) * np.abs(-0.4 * np.log(10) * sigma_mag)


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7, device='cpu'):
        """Standard Scaler for PyTorch tensors.

        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.

        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.device = device

    def fit(self, values):
        self.is_tensor = torch.is_tensor(values)
        if self.is_tensor:
            dims = list(range(values.dim() - 1))
            self.mean = torch.mean(values, dim=dims).to(self.device)
            self.std = torch.std(values, dim=dims).to(self.device)
        else:  # numpy array
            dims = list(range(values.ndim - 1))
            self.mean = np.mean(values, axis=tuple(dims))
            self.std = np.std(values, axis=tuple(dims))

    def transform(self, values, device=None):
        if device is None:
            device = self.device

        if torch.is_tensor(values):
            return (values - self.mean) / (self.std + self.epsilon).to(device)
        else:
            return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values, device=None):
        if device is None:
            device = self.device

        if torch.is_tensor(values):
            return values * Tensor(self.std).to(device) + Tensor(self.mean).to(device)
        else:
            return values * self.std + self.mean


class SpectrumPCA():
    """
    SPECULATOR PCA compression class, tensor enabled
    """

    def __init__(self, n_parameters, n_wavelengths, n_pcas, log_spectrum_filenames, parameter_selection=None):
        """
        Constructor.
        :param n_parameters: number of SED model parameters (inputs to the network)
        :param n_wavelengths: number of wavelengths in the modelled SEDs
        :param n_pcas: number of PCA components
        :param log_spectrum_filenames: list of .npy filenames for log spectra (each one an [n_samples, n_wavelengths] array)
        :param parameter_filenames: list of .npy filenames for parameters (each one an [n_samples, n_parameters] array)
        """

        # input parameters
        self.n_parameters = n_parameters
        self.n_wavelengths = n_wavelengths
        self.n_pcas = n_pcas
        self.log_spectrum_filenames = log_spectrum_filenames

        # Data scaler
        self.logspec_scaler = StandardScaler()
        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)
        # parameter selection (implementing any cuts on strange parts of parameter space)
        self.parameter_selection = parameter_selection

    def scale_spectra(self):
        # scale spectra
        log_spec = np.concatenate([np.load(self.log_spectrum_filenames[i])
                                  for i in range(len(self.log_spectrum_filenames))])
        self.logspec_scaler.fit(log_spec)
        self.normalized_logspec = self.logspec_scaler.transform(log_spec)

    # train PCA incrementally
    def train_pca(self):
        self.PCA.partial_fit(self.normalized_logspec)
        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_

    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, filename, retain=False):

        # transform the spectra to PCA basis
        training_pca = self.PCA.transform(self.normalized_logspec)

        if filename is not None:
            # save stacked transformed training data
            # the PCA coefficients
            np.save(filename + '_coeffs.npy', training_pca)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca

    def inverse_transform(self, pca_coeffs, device=None):
        # inverse transform the PCA coefficients
        if torch.is_tensor(pca_coeffs):
            assert device is not None, 'Provide device name in order to manipulate tensors'
            return torch.matmul(pca_coeffs.to(device), Tensor(self.PCA.components_).to(device)) + Tensor(self.PCA.mean_).to(device)
        else:
            return np.dot(self.PCA.components_, pca_coeffs) + self.PCA.mean_

    # make a validation plot of the PCA given some validation data
    def validate_pca_basis(self, log_spectrum_filename):

        # load in the data (and select based on parameter selection if neccessary)
        if self.parameter_selection is None:
            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)
            normalized_log_spectra = self.logspec_scaler.transform(log_spectra)
        else:
            selection = self.parameter_selection(
                np.load(self.parameter_filename))

            # load spectra and shift+scale
            log_spectra = np.load(log_spectrum_filename)[selection, :]
            normalized_log_spectra = self.logspec_scaler.transform(log_spectra)

        # transform to PCA basis and back
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = self.logspec_scaler.inverse_transform(
            self.PCA.inverse_transform(log_spectra_pca))

        # return raw spectra and spectra in basis
        return log_spectra, log_spectra_in_basis

    def save(self, filename):
        # save the PCA object
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class CustomActivation(nn.Module):
    '''
    Implementation of the activation function described in `speculator` paper: https://arxiv.org/abs/1911.11778

    ```
    $$a(\vec{x}) = [\vec{\beta} + (1 + \exp(-\vec{\alpha} \cdot \vec{x}))^{-1} (\vec{1} - \vec{\beta})] \cdot \vec{x}$$
    ```

    Shape:
        Input: (N, *) where * means, any number of additional dimensions
        Output: (N, *), same shape as the input
    Parameters:
        alphas, betas: trainable parameters
    '''

    def __init__(self, in_features):
        '''
        Initialization
            in_features: number of input features
            alphas: list of alpha values
            betas: list of beta values
        '''
        super(CustomActivation, self).__init__()
        self.in_features = in_features

        self.alphas = Parameter(torch.rand(self.in_features))
        self.betas = Parameter(torch.rand(self.in_features))

        self.alphas.requiresGrad = True  # set requiresGrad to true!
        self.betas.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        sigmoid = nn.Sigmoid()

        return torch.mul(
            (self.betas + torch.mul(
                sigmoid(
                    torch.mul(self.alphas, x)
                ), (1 - self.betas))
             ), x)


def FC(input_size, output_size):
    """Fully connect layer unit

    Args:
        input_size (int): size of input
        output_size (int): size of output

    """
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        # nn.BatchNorm1d(output_size),
        CustomActivation(output_size),
        # nn.Dropout(p=0.5)
    )


class Network(nn.Module):
    '''
    Fully connected network. Adding dropout and batch normalization will make things worse.
    '''

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.n_layers = len(hidden_size)

        for i in range(self.n_layers + 1):
            if i == 0:
                setattr(self, 'layer_' + str(i),
                        FC(input_size, hidden_size[i]))
            elif i <= self.n_layers - 1:
                setattr(self, 'layer_' + str(i),
                        FC(hidden_size[i-1], hidden_size[i]))
            else:
                setattr(self, 'layer_' + str(i),
                        FC(hidden_size[i-1], output_size))

    def forward(self, x):
        for i in range(0, self.n_layers + 1):
            x = getattr(self, 'layer_' + str(i))(x)
        return x


class Speculator():
    """
    Emulator for spectrum data. Training is done in restframe.
    """

    def __init__(self, n_parameters: int = None,
                 wavelengths=None,
                 pca_filename: str = None,
                 hidden_size: list = None):
        """
        Initialize the emulator.

        Parameters:
            n_parameters: number of parameters to be used in the emulator.
            pca_filename: filename of the PCA object to be used.
            hidden_size: list of hidden layer sizes, e.g., [100, 100, 100].
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_parameters = n_parameters  # e.g., n_parameters = 9
        self.hidden_size = hidden_size  # e.g., [100, 100, 100]
        # e.g., wavelengths = np.arange(3800, 7000, 2)
        self.wavelengths = wavelengths
        self.pca_filename = pca_filename
        with open(self.pca_filename, 'rb') as f:
            self.pca = pickle.load(f)
        self.n_pca_components = len(self.pca.pca_transform_matrix)
        self.pca_scaler = StandardScaler(device=self.device)

        self.network = Network(
            self.n_parameters, self.hidden_size, self.n_pca_components)
        self.network.to(self.device)

        self.train_loss_history = []
        self.val_loss_history = []
        
        self._build_distance_interpolator()
        
    def _build_distance_interpolator(self):
        """
        Since the `astropy.cosmology` is not differentiable, we build a distance 
        interpolator which allows us to calculate the luminosity distance at 
        any given redshift in a differentiable way.
        
        Here the cosmology is WMAP9, consistent with `prospector`.
        """
        from prospect.sources.constants import cosmo #WMAP9
        z_grid = torch.arange(0, 5, 0.005)
        dist_grid = torch.Tensor(cosmo.luminosity_distance(z_grid).value) # Mpc
        self.z_grid = z_grid.to(self.device)
        self.dist_grid = dist_grid.to(self.device)
        
    def _parse_nsa_noise_model(self, noise_model_dir):
        """
        Here we replace the scipy interpolation function with torch Interp1d.
        """
        meds_sigs, stds_sigs = np.load(noise_model_dir, allow_pickle=True)
        # meds_sigs is the median of noise, stds_sigs is the std of noise. All in magnitude.
        
        n_filters = len(meds_sigs)
        mag_grid = torch.arange(10, 30, 1)
        med_sig_grid = torch.vstack([Tensor(meds_sigs[i](mag_grid)) for i in range(n_filters)])
        std_sig_grid = torch.vstack([Tensor(stds_sigs[i](mag_grid)) for i in range(n_filters)])
        
        self.mag_grid = mag_grid.to(self.device)
        self.med_sig_grid = med_sig_grid.T.to(self.device)
        self.std_sig_grid = std_sig_grid.T.to(self.device)
        

    def load_data(self, pca_coeff, params, val_frac=0.2, batch_size=32,
                  wave_rest=torch.arange(3000, 11000, 2), wave_obs=torch.arange(3000, 11000, 2)):
        """
        Load data into the emulator.

        Parameters:
            pca_coeff: PCA coefficients of the spectrum data.
            params: parameters of the spectrum data, such as tage and tau, not include stellar mass. 
                Stellar mass is asssumed to be 1 M_sun.
            val_frac (float): fraction of the data to be used for validation.
            batch_size (int): batch size for training.
            wave_rest: restframe wavelength of the spectrum data. Default is `torch.arange(3000, 11000, 2)`.
            wave_obs: observed wavelength of the spectrum data. Default is `torch.arange(3000, 11000, 2)`.
        """
        # Normalize PCA coefficients
        self.pca_scaler.fit(pca_coeff)  # scale PCA coefficients
        pca_coeff = self.pca_scaler.transform(pca_coeff)

        assert len(pca_coeff) == len(
            params), 'PCA coefficients and parameters must have the same length'

        self.n_samples = len(pca_coeff)

        # Translate data to tensor
        x = torch.FloatTensor(params)  # physical parameters
        y = torch.FloatTensor(pca_coeff)  # PCA coefficients
        dataset = TensorDataset(x, y)

        val_len = int(self.n_samples * val_frac)
        train_len = self.n_samples - val_len

        train_data, val_data = torch.utils.data.random_split(dataset,
                                                             [train_len, val_len],
                                                             generator=torch.Generator().manual_seed(24))

        dataloaders = {}
        dataloaders['train'] = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        dataloaders['val'] = DataLoader(val_data, batch_size=val_len)

        self.dataloaders = dataloaders

        self.wave_rest = wave_rest.to(self.device)
        self.wave_obs = wave_obs.to(self.device)

    def train(self, learning_rate=0.002, n_epochs=50, display=False):
        """
        Train the NN emulator.

        Parameters:
            learning_rate (float): learning rate for the optimizer, default = 0.002.
            n_epochs (int): number of epochs for training, default = 50.
            display (bool): whether to display training/validation loss, default = False.
        """
        the_last_loss = 1e-3
        min_loss = 1e-2
        min_recon_err = 1e-3
        patience = 20
        trigger_times = 0

        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(n_epochs):
            running_loss = 0.0

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.network.train()  # Set model to training mode
                else:
                    self.network.eval()   # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = Variable(inputs).to(self.device)
                    labels = Variable(labels).to(self.device)  # .double()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.network(inputs)
                        # # Compute loss based on log-spectrum
                        # outputs = self.pca.inverse_transform(
                        #     self.pca_scaler.inverse_transform(outputs), device=self.device)
                        # labels = self.pca.inverse_transform(
                        #     self.pca_scaler.inverse_transform(labels), device=self.device)
                        loss = loss_fn(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / \
                    len(self.dataloaders[phase].dataset)
                if phase == 'train':
                    self.train_loss_history.append(epoch_loss)
                if phase == 'val':
                    self.val_loss_history.append(epoch_loss)

                if epoch % 100 == 0:
                    print(
                        'Epoch: {} - {} Loss: {:.4f}'.format(epoch, phase, epoch_loss))

            if epoch_loss > the_last_loss:
                trigger_times += 1
                #print('trigger times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return
            else:
                trigger_times = 0

            the_last_loss = epoch_loss

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                self.save_model('speculator_best_loss_model.pkl')
                self.best_loss_epoch = epoch

            for x, y in self.dataloaders['val']:
                spec = torch.log10(self.predict_spec(x))
                spec_y = self.predict_spec_from_norm_pca(y)

            recon_err = torch.nanmedian(
                torch.abs((spec - spec_y) / torch.abs(spec_y)))
            if recon_err < min_recon_err:
                min_recon_err = recon_err
                self.save_model('speculator_best_recon_model.pkl')
                self.best_recon_err_epoch = epoch

        if display:
            self.plot_loss()

    def plot_loss(self):
        """
        Plot loss curve.
        """
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        plt.plot(np.array(self.val_loss_history).flatten(), label='Val loss')
        plt.legend()

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def transform(self, spectrum_restframe, z):
        """Redshift a spectrum. Linear interpolation is used. 
        Values outside the interpolation range are linearly interpolated.

        Args:
            spectrum_restframe (torch.Tensor): restframe spectrum, shape = (n_wavelength, n_samples).
            z (torch.Tensor, or numpy array): redshifts of each spectrum, shape = (n_samples).

        Returns:
            transform function
        """
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float).to(self.device)
        z = z.squeeze()

        wave_redshifted = (self.wave_rest.unsqueeze(1) * (1 + z)).T
        
        distances = Interp1d()(self.z_grid, self.dist_grid, z)
        dfactor = ((distances * 1e5)**2 / (1 + z))
        
        # Interp1d function takes (1) the positions (`wave_redshifted`) at which you look up the value 
        # in `spectrum_restframe`, learn the interpolation function, and apply it to observation wavelengths.
        spec = Interp1d()(wave_redshifted, spectrum_restframe, self.wave_obs) / dfactor.T
        return spec

    def predict(self, params):
        """
        Predict the PCA coefficients of the spectrum, given physical parameters.
        Note: this is in restframe, and the spectrum is scaled to 1 M_sun.

        Args:
            params (torch.Tensor): physical parameters, shape = (n_samples, n_params).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        self.network.eval()
        pca_coeff = self.network(params)
        pca_coeff = self.pca_scaler.inverse_transform(
            pca_coeff, device=self.device)

        return pca_coeff

    def predict_spec(self, params, log_stellar_mass=None, redshift=None):
        """
        Predict corresponding spectrum (in linear scale), given physical parameters.

        Args:
            params (torch.Tensor): physical parameters (not including stellar mass and redshift), 
                shape = (n_samples, n_params).
            log_stellar_mass (torch.Tensor, or numpy array): log10 stellar mass of each spectrum, shape = (n_samples).
            redshift (torch.Tensor, or numpy array): redshift of each spectrum, shape = (n_samples).

        Returns:
            spec (torch.Tensor): predicted spectrum, shape = (n_wavelength, n_samples).
        """
        pca_coeff = self.predict(params)  # Assuming 1 M_sun.
        log_spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device)  # log_spectrum

        if log_stellar_mass is not None:
            if not torch.is_tensor(log_stellar_mass):
                log_stellar_mass = torch.Tensor(
                    log_stellar_mass.reshape(-1, 1)).to(self.device)
            log_spec += log_stellar_mass

        spec = 10**log_spec
        # such that interpolation will not do linear extrapolation.
        spec[:, 0] = torch.nan
        if redshift is not None:
            spec = self.transform(spec, redshift)

        spec[(redshift <= 0.0).squeeze(1)] *= 0

        return spec

    def _predict_spec_with_mass(self, params):
        """
        Predict corresponding spectrum (in linear scale), given physical parameters.

        Args:
            params (torch.Tensor): physical parameters, including mass. shape = (n_samples, n_params).
                params[:, :-1] are the physical parameters.
                params[:, -1:] is the log10 stellar mass.

        Returns:
            spec (torch.Tensor): predicted spectrum, shape = (n_wavelength, n_samples).
        """
        pca_coeff = self.predict(params[:, :-1])  # Assuming 1 M_sun and z=0.0.
        log_spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device) + params[:, -1:]  # log_spectrum

        spec = 10 ** log_spec

        return spec

    def _predict_spec_with_mass_redshift(self, params):
        """
        Predict corresponding spectrum (in linear scale), given physical parameters, stellar mass, and redshift..

        Args:
            params (torch.Tensor): physical parameters, including mass. shape = (n_samples, n_params).
                params[:, :-2] are the physical parameters (not including stellar mass and redshift).
                params[:, -2:-1] is the log10 stellar mass.
                params[:, -1:] is the redshift.

        Returns:
            spec (torch.Tensor): predicted spectrum, shape = (n_wavelength, n_samples).
        """
        pca_coeff = self.predict(params[:, :-2])  # Assuming 1 M_sun.
        log_spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device) + params[:, -2:-1]  # log_spectrum

        spec = 10 ** log_spec
        # such that interpolation will not do linear extrapolation.
        # spec[:, 0] = 0.0  # torch.nan
        spec = self.transform(spec, params[:, -1])

        return spec

    def predict_spec_from_norm_pca(self, y):
        pca_coeff = self.pca_scaler.inverse_transform(
            y.to(self.device), device=self.device)
        spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device)

        return spec

    def _calc_transmission(self, filterset):
        """Interploate transmission curves to `self.wave_obs`.
        The interpolated transmission efficiencies are saved in `self.transmission_effiency`.
        And the zeropoint counts of each filter are saved in `self.ab_zero_counts`.

        Args:
            filterset (list of string): names of filters, e.g., ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'].
        """
        x = self.wave_obs.cpu().detach().numpy()

        # transmission efficiency
        _epsilon = np.zeros((len(filterset), len(x)))
        _zero_counts = np.zeros(len(filterset))
        filters = observate.load_filters(filterset)
        for i in range(len(filterset)):
            _epsilon[i] = interp1d(filters[i].wavelength,
                                   filters[i].transmission,
                                   bounds_error=False,
                                   fill_value=0)(x)
            _zero_counts[i] = filters[i].ab_zero_counts
        self.filterset = filterset
        self.transmission_effiency = Tensor(_epsilon).to(self.device)
        self.ab_zero_counts = Tensor(_zero_counts).to(self.device)

    def predict_mag(self, params, log_stellar_mass=None, redshift=None,
                    filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
                    noise=None, noise_model_dir='./noise_model/nsa_noise_model_mag.npy', SNR=10,):
        '''
        Predict magnitudes for a given set of filters, based on the predicted spectrum. SLOW!
        See https://github.com/bd-j/prospector/blob/dda730feef5b8e679864521d0ac1c5f5f3db989c/prospect/models/sedmodel.py#L591
        and also https://github.com/pmelchior/spectrum-encoder/blob/e817fbd3bef2a3a829ea289d4b6bef41265ad60d/model.py#L255-L257.

        Parameters:
            params (torch.Tensor): physical parameters, shape = (n_samples, n_params).
            log_stellar_mass (torch.Tensor, or numpy array): log10 stellar mass of each spectrum, shape = (n_samples).
            redshift (torch.Tensor, or numpy array): redshift of each spectrum, shape = (n_samples).
            filterset (list): list of filters to predict, default = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'].
            nosie (str): whether to add noise to the predicted photometry. 
                If `noise=None`, no noise is added.
                If `noise='nsa'`, we add noise based on NSA catalog. 
                If `noise='snr'`, we add noise with constant SNR. Therefore, SNR must be provided.
            noise_model_dir (str): directory of the noise model. Only works if `noise='nsa'`.
            SNR (float): signal-to-noise ratio in maggies, default = 10 (results in ~0.1 mag noise in photometry).
                Only works if `noise='snr'`.
            
        Returns:
            mags (torch.Tensor): predicted magnitudes, shape = (n_samples, n_filters).
        '''
        if hasattr(self, 'filterset') and self.filterset != filterset:
            self.filterset = filterset
            self._calc_transmission(filterset)
        elif hasattr(self, 'filterset') and self.filterset == filterset:
            # Don't interpolate transmission efficiency again.
            pass
        elif hasattr(self, 'filterset') is False:
            self._calc_transmission(filterset)

        # get magnitude from a spectrum
        lightspeed = 2.998e18  # AA/s
        jansky_cgs = 1e-23

        _spec = self.predict_spec(params,
                                  log_stellar_mass=log_stellar_mass,
                                  redshift=redshift)

        _spec *= lightspeed / self.wave_obs**2 * \
            (3631 * jansky_cgs)  # in cgs/AA units
        _spec = torch.nan_to_num(_spec, 0.0)

        maggies = torch.trapezoid(
            ((self.wave_obs * _spec)[:, None, :] * self.transmission_effiency[None, :, :]), self.wave_obs) / self.ab_zero_counts
        
        
        if noise == 'nsa':
            ### Add noise based on NSA noise model. 
            self._parse_nsa_noise_model(noise_model_dir)
            mags = -2.5 * torch.log10(maggies) # noise-free magnitude
            
            _sigs_mags = torch.zeros_like(mags)
            _sig_flux = torch.zeros_like(mags)
            for i in range(maggies.shape[1]):
                _sigs_mags[:, i] = Interp1d()(self.mag_grid, self.med_sig_grid[:, i], mags[:, i])[0]
                _sigs_mags[:, i] += Interp1d()(self.mag_grid, self.std_sig_grid[:, i], mags[:, i])[0] * torch.randn_like(mags[:, i])
                _sig_flux[:, i] = sigma_mag2flux(_sigs_mags[:, i], mags[:, i]) # in nanomaggies
            _noise_flux = _sig_flux * torch.randn_like(mags) * 1e-9 # in maggies
            _noise_flux[(_noise_flux + maggies) < 0] = 0.0
            maggies += _noise_flux
        
        elif noise == 'snr':
            ### Add noise with constant SNR.
            _noise = torch.randn_like(maggies) * maggies / SNR
            _noise[(maggies + _noise) < 0] = 0.0
            maggies += _noise
        
        mags = -2.5 * torch.log10(maggies)
        # if noise is True:
        #     mags += torch.randn_like(mags) * 0.1
        
        return mags

    def _predict_mag_with_mass(self, params, filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz']):
        """
        Predict corresponding photometry (in magnitude), given physical parameters, stellar mass.
        Assuming redshift = 0.0.

        Args:
            params (torch.Tensor): physical parameters, including mass. shape = (n_samples, n_params).
                params[:, :-1] are the physical parameters (not including stellar mass).
                params[:, -1:] is the log10 stellar mass.
            filterset (list): list of filters to predict, default = ['sdss_{0}0'.format(b) for b in 'ugriz'].

        Returns:
            mags (torch.Tensor): predicted photometry, shape = (n_bands, n_samples).
        """
        if hasattr(self, 'filterset') and self.filterset != filterset:
            self.filterset = filterset
            self._calc_transmission(filterset)
        elif hasattr(self, 'filterset') and self.filterset == filterset:
            # Don't interpolate transmission efficiency again.
            pass
        elif hasattr(self, 'filterset') is False:
            self._calc_transmission(filterset)

        # get magnitude from a spectrum
        lightspeed = 2.998e18  # AA/s
        jansky_cgs = 1e-23

        _spec = self._predict_spec_with_mass(
            params) * lightspeed / self.wave_obs**2 * (3631 * jansky_cgs)  # in cgs/AA units

        _spec = torch.nan_to_num(_spec, 0.0)

        maggies = torch.trapezoid(
            ((self.wave_obs * _spec)[:, None, :] * self.transmission_effiency[None, :, :]), self.wave_obs) / self.ab_zero_counts
        mags = -2.5 * torch.log10(maggies)

        return mags

    def _predict_mag_with_mass_redshift(self, params, 
                                        filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
                                        noise=None, noise_model_dir='./noise_model/nsa_noise_model_mag.npy', SNR=10):
        """
        Predict corresponding photometry (in magnitude), given physical parameters, stellar mass, and redshift.

        Args:
            params (torch.Tensor): physical parameters, including mass. shape = (n_samples, n_params).
                params[:, :-2] are the physical parameters (not including stellar mass and redshift).
                params[:, -2:-1] is the log10 stellar mass.
                params[:, -1:] is the redshift.

            filterset (list): list of filters to predict, default = ['sdss_{0}0'.format(b) for b in 'ugriz'].
            nosie (str): whether to add noise to the predicted photometry. 
                If `noise=None`, no noise is added.
                If `noise='nsa'`, we add noise based on NSA catalog. 
                If `noise='snr'`, we add noise with constant SNR. Therefore, SNR must be provided.
            noise_model_dir (str): directory of the noise model. Only works if `noise='nsa'`.
            SNR (float): signal-to-noise ratio in maggies, default = 10 (results in ~0.1 mag noise in photometry).
                Only works if `noise='snr'`.

        Returns:
            mags (torch.Tensor): predicted photometry, shape = (n_bands, n_samples).
        """
        if hasattr(self, 'filterset') and self.filterset != filterset:
            self.filterset = filterset
            self._calc_transmission(filterset)
        elif hasattr(self, 'filterset') and self.filterset == filterset:
            # Don't interpolate transmission efficiency again.
            pass
        elif hasattr(self, 'filterset') is False:
            self._calc_transmission(filterset)

        # get magnitude from a spectrum
        lightspeed = 2.998e18  # AA/s
        jansky_cgs = 1e-23

        _spec = self._predict_spec_with_mass_redshift(
            params) * lightspeed / self.wave_obs**2 * (3631 * jansky_cgs)  # in cgs/AA units

        _spec = torch.nan_to_num(_spec, 0.0)

        maggies = torch.trapezoid(
            ((self.wave_obs * _spec)[:, None, :] * self.transmission_effiency[None, :, :]), self.wave_obs) / self.ab_zero_counts
        
        if noise == 'nsa':
            ### Add noise based on NSA noise model. 
            self._parse_nsa_noise_model(noise_model_dir)
            mags = -2.5 * torch.log10(maggies) # noise-free magnitude
            _sigs_mags = torch.zeros_like(mags)
            _sig_flux = torch.zeros_like(mags)
            for i in range(maggies.shape[1]):
                _sigs_mags[:, i] = Interp1d()(self.mag_grid, self.med_sig_grid[:, i], mags[:, i])[0] + Interp1d()(self.mag_grid, self.std_sig_grid[:, i], mags[:, i])[0] * torch.randn_like(mags[:, i])
                _sig_flux[:, i] = sigma_mag2flux(_sigs_mags[:, i], mags[:, i]) # in nanomaggies
            _noise = _sig_flux * torch.randn_like(mags) * 1e-9 # in maggies
            _noise[(maggies + _noise) < 0] = 0.0
            return -2.5 * torch.log10(maggies + _noise)
            
        elif noise == 'snr':
            ### Add noise with constant SNR.
            _noise = torch.randn_like(maggies) * maggies / SNR
            _noise[(maggies + _noise) < 0] = 0.0
            maggies += _noise
        
        mags = -2.5 * torch.log10(maggies)

        return mags

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)





class Photulator():
    """
    Emulator for photometry data. Directly predict magnitudes from physical parameters.
    """

    def __init__(self, n_parameters: int = None,
                 filterset: list = None,
                 hidden_size: list = None):
        """
        Initialize the emulator.

        Parameters:
            n_parameters: number of parameters to be used in the emulator.
            hidden_size: list of hidden layer sizes, e.g., [100, 100, 100].
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_parameters = n_parameters  # e.g., n_parameters = 9
        self.hidden_size = hidden_size  # e.g., [100, 100, 100]
        # e.g., wavelengths = np.arange(3800, 7000, 2)
        self.filterset = filterset

        self.n_phot = len(self.filterset)
        self.phot_scaler = StandardScaler(device=self.device)

        self.network = Network(
            self.n_parameters, self.hidden_size, self.n_phot)
        self.network.to(self.device)

        self.train_loss_history = []
        self.val_loss_history = []

    def load_data(self, phot, params, val_frac=0.2, batch_size=32):
        """
        Load data into the emulator.

        Parameters:
            phot: magnitudes in each band, not normalized.
            params: parameters of the spectrum data, such as tage and tau.
            val_frac (float): fraction of the data to be used for validation.
            batch_size (int): batch size for training.
        """
        # Normalize magnitudes
        assert phot.shape[1] == self.n_phot, 'Number of bands mismatch.'

        self.phot_scaler.fit(phot)
        phot = self.phot_scaler.transform(phot)

        assert len(phot) == len(
            params), 'magnitudes and parameters must have the same length'

        self.n_samples = len(phot)

        # Translate data to tensor
        x = torch.FloatTensor(params)  # physical parameters
        y = torch.FloatTensor(phot)  # PCA coefficients
        dataset = TensorDataset(x, y)

        val_len = int(self.n_samples * val_frac)
        train_len = self.n_samples - val_len

        train_data, val_data = torch.utils.data.random_split(dataset,
                                                             [train_len, val_len],
                                                             generator=torch.Generator().manual_seed(24))

        dataloaders = {}
        dataloaders['train'] = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        dataloaders['val'] = DataLoader(val_data, batch_size=val_len)

        self.dataloaders = dataloaders

    def train(self, learning_rate=0.002, n_epochs=50, display=False):
        """
        Train the NN emulator.

        Parameters:
            learning_rate (float): learning rate for the optimizer, default = 0.002.
            n_epochs (int): number of epochs for training, default = 50.
            display (bool): whether to display training/validation loss, default = False.
        """
        the_last_loss = 1e-3
        min_loss = 1e-2
        min_recon_err = 1e-3
        patience = 20
        trigger_times = 0

        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(n_epochs):
            running_loss = 0.0

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.network.train()  # Set model to training mode
                else:
                    self.network.eval()   # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = Variable(inputs).to(self.device)
                    labels = Variable(labels).to(self.device)  # .double()

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.network(inputs)
                        loss = loss_fn(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / \
                    len(self.dataloaders[phase].dataset)
                if phase == 'train':
                    self.train_loss_history.append(epoch_loss)
                if phase == 'val':
                    self.val_loss_history.append(epoch_loss)

                if epoch % 100 == 0:
                    print(
                        'Epoch: {} - {} Loss: {:.4f}'.format(epoch, phase, epoch_loss))

            if epoch_loss > the_last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return
            else:
                trigger_times = 0

            the_last_loss = epoch_loss

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                self.save_model('phot_best_loss_model.pkl')
                self.best_loss_epoch = epoch

            for x, y in self.dataloaders['val']:
                phot_pre = self.predict_mag(x)  # x is params, y is magnitudes
                phot_val = self.phot_scaler.inverse_transform(
                    y.to(self.device))

            recon_err = torch.median(
                torch.abs((phot_pre - phot_val) / torch.abs(phot_val)))
            if recon_err < min_recon_err:
                min_recon_err = recon_err
                self.save_model('phot_best_recon_err_model.pkl')
                self.best_recon_err_epoch = epoch

        if display:
            self.plot_loss()

    def plot_loss(self):
        """
        Plot loss curve.
        """
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        plt.plot(np.array(self.val_loss_history).flatten(), label='Val loss')
        plt.legend()

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def predict_mag(self, params):
        """
        Predict magnitudes, given physical parameters.
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        self.network.eval()
        phot = self.network(params)
        phot = self.phot_scaler.inverse_transform(
            phot, device=self.device)

        return phot

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
