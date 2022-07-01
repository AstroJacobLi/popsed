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

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

from sedpy import observate

from tqdm import trange

from . import utils


class StandardScaler:
    """
    Standard scaler for z-scoring data.

    I made this because the sklearn scaler is not compatible with PyTorch tensors.
    """

    def __init__(self, mean=None, std=None, epsilon=1e-7, device='cpu'):
        """Standard Scaler for PyTorch tensors.

        The class can be used to normalize PyTorch Tensors using native functions.
        The module does not expect the tensors to be of any specific shape;
        as long as the features are the last dimension in the tensor,
        the module will work fine.

        Parameters
        ----------
        mean: float. The mean of the features.
            The property will be set after a call to fit.
        std: float. The standard deviation of the features.
            The property will be set after a call to fit.
        epsilon: float. Used to avoid a Division-By-Zero exception.
        device: str. The device to use for the tensors, either 'cpu' or 'cuda'.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.device = device

    def fit(self, values):
        """
        Fit the StandardScaler to the data. i.e. calculate
        the mean and standard deviation of the data.

        Parameters
        ----------
        values: torch.Tensor, or np.array. The data to fit the scaler to.
        """
        self.is_tensor = torch.is_tensor(values)
        if self.is_tensor:
            dims = list(range(values.dim() - 1))
            goodflag = ~(torch.isnan(values).any(dim=-1) |
                         torch.isinf(values).any(dim=-1))
            self.mean = torch.mean(values[goodflag], dim=dims).to(self.device)
            self.std = torch.std(values[goodflag], dim=dims).to(self.device)
        else:  # numpy array
            dims = list(range(values.ndim - 1))
            self.mean = np.nanmean(values, axis=tuple(dims))
            self.std = np.nanstd(values, axis=tuple(dims))

    def transform(self, values, device=None):
        """
        Transform the input data to be z-scored, based on
        the mean and standard deviation calculated during the fit.

        Parameters
        ----------
        values: torch.Tensor, or np.array. The data to be transformed.
        device: str. The device to use for the tensors, either 'cpu' or 'cuda'.

        Returns
        -------
        torch.Tensor. The transformed data.
        """
        if device is None:
            device = self.device

        if torch.is_tensor(values):
            return ((values - self.mean.to(device)) / (self.std.to(device) + self.epsilon)).to(device)
        else:
            return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values, device=None):
        """
        Inverse transform the input data to be un-z-scored, based on
        the mean and standard deviation calculated during the fit.

        Parameters
        ----------
        values: torch.Tensor, or np.array. The data to be inverse transformed.
        device: str. The device to use for the tensors, either 'cpu' or 'cuda'.

        Returns
        -------
        torch.Tensor. The inverse transformed data.
        """
        if device is None:
            device = self.device

        if torch.is_tensor(values):
            return values * Tensor(self.std).to(device) + Tensor(self.mean).to(device)
        else:
            return values * self.std + self.mean


class SpectrumPCA():
    """
    A PCA class designed for compressing the spectra into a lower dimensional space.

    Reference:
        - https://arxiv.org/abs/1911.11778
        - https://github.com/justinalsing/speculator
    """

    def __init__(self, n_pcas, log_spectrum_filenames, parameter_selection=None):
        """
        Initialize the PCA class, based on `sklearn.decomposition.IncrementalPCA`.
        We use IncrementalPCA because the training set is huge (~10^6 spectra).

        Parameters
        ----------
        n_pcas: int. The number of principal components to use.
        log_spectrum_filenames: list of .npy filenames for log spectra
            (each one is an [n_samples, n_wavelengths] array)
        parameter_filenames: list of .npy filenames for parameters
            (each one is an [n_samples, n_parameters] array)
        """
        self.n_pcas = n_pcas
        self.log_spectrum_filenames = log_spectrum_filenames
        # Data scaler. The log_spectra need to be z-scored before being fed to the PCA.
        self.logspec_scaler = StandardScaler()
        # PCA object
        self.PCA = IncrementalPCA(n_components=self.n_pcas)

    def scale_spectra(self):
        """
        The log spectra need to be z-scored before being fed to the PCA.
        This function first train a scaler, then z-score the log spectra.
        """
        log_spec = np.concatenate([np.load(self.log_spectrum_filenames[i])
                                  for i in range(len(self.log_spectrum_filenames))])
        log_spec = utils.interp_nan(log_spec)

        self.logspec_scaler.fit(log_spec)
        self.normalized_logspec = self.logspec_scaler.transform(log_spec)

    def train_pca(self, chunk_size=1000):
        """
        Train the PCA model incrementally. Because of the large size of
        training set, we fit the PCA model in chunks.

        Parameters
        ----------
        chunk_size: int. The number of spectra used in each step.
        """
        _chunks = list(utils.split_chunks(self.normalized_logspec, chunk_size))
        t = trange(len(_chunks),
                   desc='Training PCA model',
                   unit='chunks')
        for i in t:
            self.PCA.partial_fit(_chunks[i])

        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_

    def inverse_transform(self, pca_coeffs, device=None):
        """
        Inverse transform the PCA coefficients to get the original log spectra.

        Parameters
        ----------
        pca_coeffs: torch.Tensor. The PCA coefficients.
        device: str. The device to use for the tensors, either 'cpu' or 'cuda'.

        Returns
        -------
        torch.Tensor. The corresponding log spectra.
        """
        # inverse transform the PCA coefficients
        if torch.is_tensor(pca_coeffs):
            assert device is not None, 'Provide device name in order to manipulate tensors'
            return torch.matmul(pca_coeffs.to(device),
                                Tensor(self.PCA.components_).to(device)
                                ) + Tensor(self.PCA.mean_).to(device)
        else:
            return np.dot(self.PCA.components_, pca_coeffs) + self.PCA.mean_

    def _transform_and_stack_training_data(self, filename, retain=False):
        """
        Transform the training data set to PCA basis.

        I actually don't use this function.
        """
        # transform the spectra to PCA basis
        training_pca = self.PCA.transform(self.normalized_logspec)

        if filename is not None:
            # save stacked transformed training data
            # the PCA coefficients
            np.save(filename + '_coeffs.npy', training_pca)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca

    def validate_pca_basis(self, log_spectrum_filename):
        """
        Test PCA reconstruction accuracy on a validation set.

        Parameters
        ----------
        log_spectrum_filename: str. The filename of the validation set.

        Returns
        -------
        log_spectra: the input log spectra.
        log_spectra_in_basis: the reconstructed log spectra.
        """
        # load in the data (and select based on parameter selection if neccessary)
        # if self.parameter_selection is None:
        # load spectra and shift+scale
        log_spectra = np.load(log_spectrum_filename)
        log_spectra = utils.interp_nan(log_spectra)
        normalized_log_spectra = self.logspec_scaler.transform(log_spectra)
        # else:
        #     selection = self.parameter_selection(
        #         np.load(self.parameter_filename))

        #     # load spectra and shift+scale
        #     log_spectra = np.load(log_spectrum_filename)[selection, :]
        #     log_spectra = utils.interp_nan(log_spectra)
        #     normalized_log_spectra = self.logspec_scaler.transform(log_spectra)

        # transform to PCA basis and back
        log_spectra_pca = self.PCA.transform(normalized_log_spectra)
        log_spectra_in_basis = self.logspec_scaler.inverse_transform(
            self.PCA.inverse_transform(log_spectra_pca))

        # return raw spectra and spectra in basis
        return log_spectra, log_spectra_in_basis

    def save(self, filename):
        """
        Save PCA model as pickle file.

        Parameters
        ----------
        filename: str. The filename to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class CustomActivation(nn.Module):
    '''
    Implementation of the activation function described in the `speculator` paper: https://arxiv.org/abs/1911.11778

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

    Parameters
    ----------
    input_size (int): size of input
    output_size (int): size of output

    Returns
    -------
    a sequential fully connected layer
    """
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        # nn.BatchNorm1d(output_size),
        CustomActivation(output_size),
        # nn.Dropout(p=0.5)
    )


class Network(nn.Module):
    '''
    Fully connected network.

    WARNING: **Adding dropout and batch normalization will make things worse.**
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
                        FC(hidden_size[i - 1], hidden_size[i]))
            else:
                setattr(self, 'layer_' + str(i),
                        FC(hidden_size[i - 1], output_size))

    def forward(self, x):
        for i in range(0, self.n_layers + 1):
            x = getattr(self, 'layer_' + str(i))(x)
        return x


class Speculator():
    """
    An emulator for SPS model.
    """
    from .models import lightspeed, to_cgs_at_10pc, jansky_cgs

    def __init__(self, name='NMF', model='NMF',
                 n_parameters: int = None,
                 pca_filename: str = None,
                 hidden_size: list = [256, 256, 256, 256]):
        """
        Initialize the emulator.

        The emulator takes SPS physical parameters as input, and returns the
        corresponding rest-frame spectra. In some SPS models, the redshift is
        needed in training because it sets the length of non-parametric SFH.
        After training, the spectra can be obtained for any SPS parameters.

        Parameters
        ----------
        name: str. Name of the emulator.
        n_parameters: int. Number of parameters to be used in the emulator.
        pca_filename: int. Filename of the PCA object to be used.
        hidden_size: list. List of hidden layer sizes, e.g., [100, 100, 100].

        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # default is GPU
        self.name = name
        self._model = model
        self.n_parameters = n_parameters  # e.g., n_parameters = 9
        self.hidden_size = hidden_size  # e.g., [256, 256, 256, 256]
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
        self._build_params_prior()

    def _build_params_prior(self):
        """
        Hard bound prior for the input physical parameters.
        E.g., redshift cannot be negative.

        WARNING:
        / This prior should be consistant with the \
        \ prior used in training the emulator.     /
        ----------------------------------------
                \   ^__^
                \  (oo)\_______
                    (__)\       )\/\
                        ||----w |
                        ||     ||
        """
        if self._model == 'tau':
            self.prior = {'tage': [0, 14],
                          'logtau': [-4, 4],
                          'logzsol': [-3, 2],
                          'dust2': [0, 5],
                          'logm': [0, 16],
                          'redshift': [0, 10]}
        elif self._model == 'NMF':
            self.prior = {'beta1_sfh': [0, 1], 'beta2_sfh': [0, 1],
                          'beta3_sfh': [0, 1], 'beta4_sfh': [0, 1],
                          'fburst': [0, 1.0], 'tburst': [1e-2, 13.27],
                          'logzsol': [-2.6, 0.3],
                          'dust1': [0, 3], 'dust2': [0, 3], 'dust_index': [-3, 1],
                          'logm': [0, 16],
                          'redshift': [0, 1.5]}

    def _build_distance_interpolator(self):
        """
        Since the `astropy.cosmology` is not differentiable, we build a distance
        interpolator which allows us to calculate the luminosity distance at
        any given redshift in a differentiable way.

        Here the cosmology is Planck15, NOT consistent with `prospector`.
        """
        from astropy.cosmology import Planck15 as cosmo
        z_grid = torch.arange(0, 5, 0.001)
        dist_grid = torch.Tensor(
            cosmo.luminosity_distance(z_grid).value)  # Mpc
        self.z_grid = z_grid.to(self.device)
        self.dist_grid = dist_grid.to(self.device)

    def _parse_nsa_noise_model(self, noise_model_dir):
        """
        Parse the noise model from the NSA.
        The noise model is generated in `popsed/notebook/forward_model/noise_model/``.

        Parameters
        ----------
        noise_model_dir: str. The directory of the noise model file.
        """
        meds_sigs, stds_sigs = np.load(noise_model_dir, allow_pickle=True)
        # meds_sigs is the median of noise, stds_sigs is the std of noise. All in magnitude.

        n_filters = len(meds_sigs)
        mag_grid = torch.arange(10, 30, 1)
        med_sig_grid = torch.vstack(
            [Tensor(meds_sigs[i](mag_grid)) for i in range(n_filters)])
        std_sig_grid = torch.vstack(
            [Tensor(stds_sigs[i](mag_grid)) for i in range(n_filters)])

        self.mag_grid = mag_grid.to(self.device)
        self.med_sig_grid = med_sig_grid.T.to(self.device)
        self.std_sig_grid = std_sig_grid.T.to(self.device)

    def load_data(self, pca_coeff, params,
                  params_name=['tage', 'logtau', 'logm', 'redshift'],
                  val_frac=0.2, batch_size=512,
                  wave_rest=torch.arange(3000, 11000, 2),
                  wave_obs=torch.arange(3000, 11000, 2)):
        """
        Load training data into the emulator.

        Parameters
        ----------
        pca_coeff: np.ndarray.
            PCA coefficients of the training spectra, shape = (n_samples, n_pca).
            This is generated using the SpectrumPCA class.
        params: np.ndarray.
            Parameters of the training spectra, shape = (n_samples, n_parameters).
            E.g., (beta1_sfh, beta2_sfh, beta3_sfh, beta4_sfh, fburst, tburst,
                   logzsol, dust1, dust2, dust_index, redshift)
            NOTE: stellar mass should not be included, because all stellar mass
                  is asssumed to be 1 M_sun. But redshift (as a proxy of t_age
                  for NMF-based SFH) should be included.
        val_frac: float.
            Fraction of the training data to be used for validation.
        batch_size: int.
            Batch size for training. Larger batch size will speed up training
            but require more memory and larger learning rates.
            The current best batch size is 512.
        wave_rest: torch.Tensor. Restframe wavelength of the input spectra, as a tensor.
        wave_obs: torch.Tensor. Observed wavelength of the input spectra, as a tensor.
        """
        self.params_name = params_name

        # Normalize PCA coefficients
        self.pca_scaler.fit(pca_coeff)  # scale PCA coefficients
        pca_coeff = self.pca_scaler.transform(pca_coeff).astype(np.float32)

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
                                                             generator=torch.Generator().manual_seed(42))

        dataloaders = {}
        dataloaders['train'] = DataLoader(
            train_data, batch_size=batch_size, shuffle=True)
        dataloaders['val'] = DataLoader(
            val_data, batch_size=batch_size, shuffle=True)

        self.dataloaders = dataloaders

        self.wave_rest = wave_rest.to(self.device)
        self.wave_obs = wave_obs.to(self.device)

        self.bounds = np.array([self.prior[key] for key in self.params_name])

    def train(self,
              learning_rate=1e-3,
              n_epochs=50,
              display=False,
              scheduler=None,
              scheduler_args=None):
        """
        Train the network in emulator, using Adam optimizer,
        and MSE loss on the z-scored log10 spec (not on PCA coeffs).

        Parameters
        ----------
        learning_rate: float. Learning rate for the optimizer, default = 1e-3.
        n_epochs: int. Number of epochs for training, default = 50.
        display: bool. Whether to display training and validation loss, default = False.
        scheduler: torch.optim.lr_scheduler. Learning rate scheduler, default = None.
        scheduler_args: dict. Arguments for the learning rate scheduler, default = None.
        """

        the_last_loss = 1e-3
        min_loss = 1e-2
        min_recon_err = 2
        patience = 20
        trigger_times = 0

        self.optimizer = optim.Adam(self.network.parameters())

        # Config the learning rate scheduler
        if scheduler is not None and scheduler_args is not None:
            scheduler = scheduler(self.optimizer, **scheduler_args)
        else:
            self.optimizer = optim.Adam(
                self.network.parameters(), lr=learning_rate)

        loss_fn = nn.MSELoss()

        t = trange(n_epochs,
                   desc='Training Speculator',
                   unit='epochs')

        for epoch in t:
            for phase in ['train', 'val']:
                running_loss = 0.0  # the accumulative loss in one epoch
                if phase == 'train':
                    self.network.train()  # Set model to training mode
                else:
                    self.network.eval()   # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = Variable(inputs).to(self.device)
                    labels = Variable(labels).to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.network(inputs)

                        # Compute loss based on the z-scored log-spectrum
                        outputs = self.pca.inverse_transform(
                            self.pca_scaler.inverse_transform(outputs), device=self.device)
                        labels = self.pca.inverse_transform(
                            self.pca_scaler.inverse_transform(labels), device=self.device)

                        loss = loss_fn(outputs, labels)

                        # backward + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            if scheduler is not None:
                                scheduler.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(self.dataloaders[phase])
                if phase == 'train':
                    self.train_loss_history.append(epoch_loss)
                if phase == 'val':
                    self.val_loss_history.append(epoch_loss)

            if self.train_loss_history[-1] > the_last_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return
            else:
                trigger_times = 0

            the_last_loss = epoch_loss

            if self.train_loss_history[-1] < min_loss:
                min_loss = self.train_loss_history[-1]
                self.save_model('speculator_best_loss_model.pkl')
                self.best_loss_epoch = len(self.train_loss_history) - 1

            for x, y in self.dataloaders['val']:
                spec = torch.log10(self.predict_spec(x))
                spec_y = self.predict_spec_from_norm_pca(y)

            recon_err = torch.nanmedian(
                torch.abs((10**spec - 10**spec_y) / torch.abs(10**spec_y))) * 100
            if recon_err < min_recon_err:
                min_recon_err = recon_err
                self.save_model('speculator_best_recon_model.pkl')
                self.best_recon_err_epoch = len(self.train_loss_history) - 1

            t.set_description(
                f'Loss = {self.train_loss_history[-1]:.6f} (train), {self.val_loss_history[-1]:.6f} (val), {recon_err.item():.6f} (recon_err)')

        if display:
            self.plot_loss()

        print(
            'Epoch: {} - {} Train Loss: {:.6f}'.format(
                len(self.train_loss_history), phase, self.train_loss_history[-1])
        )
        print(
            'Epoch: {} - {} Vali Loss: {:.6f}'.format(
                len(self.val_loss_history), phase, self.val_loss_history[-1])
        )
        print(f'Recon error: {recon_err.item():.6f}')
        if scheduler is not None:
            print('lr:', scheduler.get_lr())

    def plot_loss(self):
        """
        Plot the loss curve.
        """
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        plt.plot(np.array(self.val_loss_history).flatten(), label='Vali loss')
        plt.legend()

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def transform(self, spectra_restframe, z, islog=True):
        """
        Redshift spectra.
        Linear interpolation is used. Values outside
        the interpolation range are linearly interpolated.

        Parameters
        ----------
        spectra_restframe: torch.Tensor.
            Restframe spectra in **linear** flux, shape = (n_wavelength, n_samples).
        z: torch.Tensor or np.ndarray.
            Redshifts of each spectra, shape = (n_samples,).
        islog: bool.
            Whether the input spectra is in log10-flux.
            If True, the output spectra is also in log10-flux.

        Returns
        -------
        spectra_transformed: torch.Tensor. Redshifted spectra.
        """
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float).to(self.device)
        z = z.squeeze()

        if torch.any(z > 0):
            wave_redshifted = (self.wave_rest.unsqueeze(1) * (1 + z)).T

            distances = Interp1d()(self.z_grid, self.dist_grid, z)
            # 1e5 because the absolute mag is 10pc.
            dfactor = ((distances * 1e5)**2 / (1 + z))

            # Interp1d function takes (1) the positions (`wave_redshifted`) at which you look up the value
            # in `spectrum_restframe`, learn the interpolation function, and apply it to observation wavelengths.
            if islog:
                spec = Interp1d()(wave_redshifted, spectra_restframe,
                                  self.wave_obs) - torch.log10(dfactor.T)
            else:
                spec = Interp1d()(wave_redshifted, spectra_restframe, self.wave_obs) / dfactor.T
            return spec
        else:
            return spectra_restframe

    def predict(self, params):
        """
        Predict the PCA coefficients of the spectrum, given the SPS physical parameters.
        Note: this is in restframe, and the spectrum is scaled to 1 M_sun.

        Parameters
        ----------
        params (torch.Tensor): SPS physical parameters, shape = (n_samples, n_params).

        Returns
        -------
        pca_coeff (torch.Tensor): PCA coefficients, shape = (n_samples, n_pca_coeffs).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        self.network.eval()
        pca_coeff = self.network.to(self.device)(params)
        pca_coeff = self.pca_scaler.inverse_transform(
            pca_coeff, device=self.device)

        return pca_coeff

    def predict_spec(self, params, log_stellar_mass=None, redshift=None):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.

        Parameters
        ----------
        params: torch.Tensor. The SPS physical parameters, **not including stellar mass and redshift**.
            shape = (n_samples, n_params).
        log_stellar_mass: torch.Tensor or np.ndarray. log10 stellar mass of each spectrum, shape = (n_samples).
        redshift: torch.Tensor or np.ndarray. Redshift of each spectrum, shape = (n_samples).

        Returns
        -------
        spec: torch.Tensor. The predicted spectra, shape = (n_wavelength, n_samples).
            **Notice** that if `self._model == "NMF"`, the output spectra is in unit of Lsun/Hz.
            To convert from L_sun/Hz to L_sun/AA, multiply L_sun/Hz by lightspeed / wave**2.
            To convert L_sun/Hz to erg/s/cm^2/AA at 10 pc, multiply by `to_cgs_at_10pc`.
        """
        if log_stellar_mass is None:
            log_stellar_mass = torch.zeros_like(params[:, 0:1])
        if redshift is None:
            redshift = torch.zeros_like(params[:, 0:1])
        return self._predict_spec_with_mass_redshift(torch.hstack([params, log_stellar_mass, redshift]).to(self.device))

    def _predict_spec_with_mass_redshift(self, params):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.

        Parameters
        ----------
        params: torch.Tensor.
            SPS physical parameters, including stellar mass. shape = (n_samples, n_params).
            params[:, :-2] are the SPS physical parameters NOT including stellar mass and redshift.
                If you are using non-parametric SPS model, you might include redshift (as a proxy for t_age).
            params[:, -2:-1] is the log10 stellar mass.
            params[:, -1:] is the redshift, used to shift and dim the spectra.

        Returns
        -------
        spec: torch.Tensor.
            Predicted spectra in linear scales, shape = (n_wavelength, n_samples).
            **Notice** that if `self._model == "NMF"`, the output spectra is in unit of Lsun/Hz.
            To convert from L_sun/Hz to L_sun/AA, multiply L_sun/Hz by lightspeed / wave**2.
            To convert L_sun/Hz to erg/s/cm^2/AA at 10 pc, multiply by `to_cgs_at_10pc`.
        """
        pca_coeff = self.predict(params[:, :-2])  # Assuming 1 M_sun.
        log_spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device) + params[:, -2:-1]  # added log_stellar_mass

        thresh = 5
        # if torch.any(log_spec > thresh):
        # print(f'# of log spec > {thresh} params:',
        #       (log_spec > thresh).any(dim=1).sum())
        log_spec[torch.any(log_spec > thresh, dim=1)] = -30
        # log_spec[torch.any(log_spec > 20, dim=1)] = -12
        # such that interpolation will not do linear extrapolation.
        # spec[:, 0] = 0.0  # torch.nan
        # spec = self.transform(10**log_spec, params[:, -1:], islog=False)
        spec = 10**self.transform(log_spec, params[:, -1], islog=True)
        spec = 10**log_spec
        # if torch.any(torch.isnan(spec)):
        #     print(params[torch.isnan(spec).any(dim=1)])
        # print('Nan in spec:', torch.isnan(spec).sum())
        # print('Correpsonding spec:', log_spec[torch.isnan(spec).any(dim=1)])
        # I don't directly ban unphysical spectra here. But I add penalty term to the loss function.
        # bad_mask = torch.stack([((params < self.bounds[i][0]) | (params > self.bounds[i][1]))[
        #     :, i] for i in range(len(self.bounds))]).sum(dim=0, dtype=bool)
        # bad_val = 1e-12  # -torch.inf
        # spec[bad_mask] = bad_val
        # print('Bad mask:', bad_mask.sum())
        # spec[(params[:, -1:] < 0.0).squeeze(1)] = bad_val
        # spec[(params[:, -2:-1] < 0.0).squeeze(1)] = bad_val
        return spec

    def _predict_spec_with_mass_restframe(self, params):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.

        Parameters
        ----------
        params: torch.Tensor.
            SPS physical parameters, including stellar mass. shape = (n_samples, n_params).
            params[:, :-1] are the SPS physical parameters NOT including stellar mass and redshift.
                If you are using non-parametric SPS model, you might include redshift (as a proxy for t_age).
            params[:, -1:] is the log10 stellar mass.

        Returns
        -------
        spec: torch.Tensor.
            Predicted spectra in linear scales, shape = (n_wavelength, n_samples).
            **Notice** that if `self._model == "NMF"`, the output spectra is in unit of Lsun/Hz.
            To convert from L_sun/Hz to L_sun/AA, multiply L_sun/Hz by lightspeed / wave**2.
            To convert L_sun/Hz to erg/s/cm^2/AA at 10 pc, multiply by `to_cgs_at_10pc`.
        """
        pca_coeff = self.predict(params[:, :-1])  # Assuming 1 M_sun.
        log_spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device) + params[:, -1:]  # added log_stellar_mass

        thresh = 5
        # if torch.any(log_spec > thresh):
        #     print(f'# of log spec > {thresh} params:',
        #           (log_spec > thresh).any(dim=1).sum())
        log_spec[torch.any(log_spec > thresh, dim=1)] = -30
        spec = 10**log_spec
        return spec

    def predict_spec_from_norm_pca(self, y):
        pca_coeff = self.pca_scaler.inverse_transform(
            y.to(self.device), device=self.device)
        spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device)

        return spec

    def _calc_transmission(self, filterset, filter_dir=None):
        import sys
        sys.path.append('/home/jiaxuanl/Research/Packages/sedpy/')

        """
        Interploate and evaluate transmission curves at `self.wave_obs`.
        Also calculate the zeropoint in each filter.
        The interpolated transmission efficiencies are saved in `self.transmission_effiency`.
        And the zeropoint counts of each filter are saved in `self.ab_zero_counts`.

        Parameters
        ----------
        filterset: list of strings.
            Names of filters, e.g., ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'].
            You can look up at available filters using `sedpy.observate.list_available_filters`.
        """
        x = self.wave_obs.cpu().detach().numpy()

        # transmission efficiency
        _epsilon = np.zeros((len(filterset), len(x)))
        _zero_counts = np.zeros(len(filterset))
        if filter_dir is None:
            filters = observate.load_filters(filterset)
        else:
            filters = observate.load_filters(filterset, directory=filter_dir)

        for i in range(len(filterset)):
            _epsilon[i] = interp1d(filters[i].wavelength,
                                   filters[i].transmission,
                                   bounds_error=False,
                                   fill_value=0)(x)
            _zero_counts[i] = filters[i].ab_zero_counts
        self.filterset = filterset
        self.filter_dir = filter_dir
        self.transmission_effiency = Tensor(_epsilon).to(self.device)
        self.ab_zero_counts = Tensor(_zero_counts).to(self.device)

    def predict_mag(self, params, log_stellar_mass=None, redshift=None, **kwargs):
        """
        Predict the corresponding magnitude for given physical parameters.

        Parameters
        ----------
        params: torch.Tensor. The SPS physical parameters, **not including stellar mass and redshift**.
            shape = (n_samples, n_params).
        log_stellar_mass: torch.Tensor or np.ndarray. log10 stellar mass of each spectrum, shape = (n_samples).
        redshift: torch.Tensor or np.ndarray. Redshift of each spectrum, shape = (n_samples).
        kwargs: you can pass filterset and noise model here. See `self._predict_mag_with_mass_redshift`.

        Returns
        -------
        mag: torch.Tensor. The predicted magnitudes, shape = (n_wavelength, n_samples).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        if log_stellar_mass is None:
            log_stellar_mass = torch.zeros_like(params[:, 0:1])
        if redshift is None:
            redshift = torch.zeros_like(params[:, 0:1])
        return self._predict_mag_with_mass_redshift(torch.hstack([params, log_stellar_mass, redshift]).to(self.device), **kwargs)

    def _predict_mag_with_mass_redshift(self, params,
                                        filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
                                        noise=None, noise_model_dir='./noise_model/nsa_noise_model_mag.npy', SNR=10):
        """
        Predict corresponding photometry (in magnitude), given SPS physical parameters.

        Parameters
        ----------
        params: torch.Tensor.
            SPS physical parameters, including stellar mass. shape = (n_samples, n_params).
            params[:, :-2] are the SPS physical parameters NOT including stellar mass and redshift.
                If you are using non-parametric SPS model, you might include redshift (as a proxy for t_age).
            params[:, -2:-1] is the log10 stellar mass.
            params[:, -1:] is the redshift, used to shift and dim the spectra.

        filterset: list.
            List of filters to predict photometry, default = ['sdss_{0}0'.format(b) for b in 'ugriz'].
        nosie: str.
            Whether to add noise to the predicted photometry.
            If `noise=None`, no noise is added.
            If `noise='nsa'`, we add noise based on NSA catalog.
            If `noise='snr'`, we add noise with constant SNR. Therefore, SNR must be provided.
        noise_model_dir: str.
            The directory of the noise model. Only works if `noise='nsa'`.
        SNR: float.
            The signal-to-noise ratio in maggies, default = 10 (results in ~0.1 mag noise in photometry).
            Only works if `noise='snr'`.

        Returns:
            mags: torch.Tensor.
                Predicted photometry, shape = (n_bands, n_samples).
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
        # Note that if the SPS model is NMF, the output spectra from `self._predict_spec_with_mass_redshift`
        # are in unit of Lsun/Hz dimmed by `dfactor` in `self.transform`. We need to convert
        # to Lsun/AA, then to erg/s/AA at 10pc.
        # If SPS model is tau, no problem exists.

        if self._model == 'NMF':
            _spec = self._predict_spec_with_mass_redshift(
                params) * self.lightspeed / self.wave_obs**2 * self.to_cgs_at_10pc
        elif self._model == 'tau':
            _spec = self._predict_spec_with_mass_redshift(
                params) * self.lightspeed / self.wave_obs**2 * (3631 * self.jansky_cgs)  # in cgs/AA units
        else:
            raise NotImplementedError(
                'SPS model {} is not implemented.'.format(self._model))

        _spec = torch.nan_to_num(_spec, 0.0)

        maggies = torch.trapezoid(
            ((self.wave_obs * _spec)[:, None, :] * self.transmission_effiency[None, :, :]
             ), self.wave_obs) / self.ab_zero_counts

        if noise == 'nsa':
            # Add noise based on NSA noise model.
            self._parse_nsa_noise_model(noise_model_dir)
            mags = -2.5 * torch.log10(maggies)  # noise-free magnitude
            _sigs_mags = torch.zeros_like(mags)
            _sig_flux = torch.zeros_like(mags)
            for i in range(maggies.shape[1]):
                _sigs_mags[:, i] = Interp1d()(self.mag_grid, self.med_sig_grid[:, i], mags[:, i])[
                    0] + Interp1d()(self.mag_grid, self.std_sig_grid[:, i], mags[:, i])[0] * torch.randn_like(mags[:, i])
                _sig_flux[:, i] = utils.sigma_mag2flux(
                    _sigs_mags[:, i], mags[:, i])  # in nanomaggies
            _noise = _sig_flux * torch.randn_like(mags) * 1e-9  # in maggies
            _noise[(maggies + _noise) < 0] = 0.0
            return -2.5 * torch.log10(maggies + _noise)

        elif noise == 'snr':
            # Add noise with constant SNR.
            _noise = torch.randn_like(maggies) * maggies / SNR
            _noise[(maggies + _noise) < 0] = 0.0
            maggies += _noise

        if torch.isnan(maggies).any() or torch.isinf(maggies).any():
            print(maggies)
        mags = -2.5 * torch.log10(maggies)

        return mags

    def save_model(self, filename):
        """
        Save the emulator to pickle file.

        Parameters
        ----------
        filename: str. The filename of the pickle file.
        """
        with open(filename.replace('.pkl', '') + '_' + self.name + '.pkl', 'wb') as f:
            pickle.dump(self, f)


class SuperSpeculator():
    """
    A class to combine different speculators trained for certain wavelengths.
    """

    from .models import lightspeed, to_cgs_at_10pc

    def __init__(self, speculators_dir=None, str_wbin=[
        '.w1000_2000',
        '.w2000_3600',
        '.w3600_5500',
        '.w5500_7410',
        '.w7410_60000'
    ], wavelength=None, params_name=None, device='cuda'):
        """
        Initialize the SuperSpeculator.

        Parameters
        ----------
        speculators_dir: str, the directory of each speculator.
        str_wbin: list of str. The wavelength bin of each speculator.
        wavelength: list of float. The wavelength of each speculator. 
            The wavelength is stored at './train_sed_NMF/nmf_seds/fsps.wavelength.npy'
        params_name: list of str. The name of the parameters of each speculator. 
            The default would be ['kappa1_sfh', 'kappa2_sfh', 'kappa3_sfh', 
                 'fburst', 'tburst', 'logzsol', 'dust1', 'dust2', 
                 'dust_index', 'redshift', 'logm']. 
            Please follow the order such that redshift is the second last one, 
            and log stellar mass is the last one.
        device: str. The device to run the model.

        """
        speculators = []
        for file in speculators_dir:
            with open(file, 'rb') as f:
                speculators.append(pickle.load(f))

        for _speculator in speculators:
            _speculator.device = device
            _speculator.network.eval()
            assert _speculator._model == 'NMF', 'Only NMF model is supported.'

        self._model = 'NMF'
        self.speculators = speculators
        self.str_wbin = str_wbin
        self.device = device
        self.wavelength = torch.Tensor(wavelength).to(self.device)
        self.params_name = params_name
        self._build_distance_interpolator()
        self._build_params_prior()

    def _build_distance_interpolator(self):
        """
        Since the `astropy.cosmology` is not differentiable, we build a distance
        interpolator which allows us to calculate the luminosity distance at
        any given redshift in a differentiable way.

        Here the cosmology is Planck15, NOT consistent with `prospector`.
        """
        from astropy.cosmology import Planck15 as cosmo
        z_grid = torch.arange(0, 5, 0.001)
        dist_grid = torch.Tensor(
            cosmo.luminosity_distance(z_grid).value)  # Mpc
        self.z_grid = z_grid.to(self.device)
        self.dist_grid = dist_grid.to(self.device)

    def _build_params_prior(self):
        """
        Hard bound prior for the input physical parameters.
        E.g., redshift cannot be negative. 
        We replace zero with 1e-10 such that its log is not -inf.

        WARNING:
        / This prior should be consistant with the \
        \ prior used in training the emulator.     /
        ----------------------------------------
                \   ^__^
                \  (oo)\_______
                    (__)\       )\/\
                        ||----w |
                        ||     ||
        """
        if self._model == 'tau':
            self.prior = {'tage': [0, 14],
                          'logtau': [-4, 4],
                          'logzsol': [-3, 2],
                          'dust2': [0, 5],
                          'logm': [0, 16],
                          'redshift': [0, 10]}
        elif self._model == 'NMF':
            self.prior = {'kappa1_sfh': [1e-10, 1],
                          'kappa2_sfh': [1e-10, 1],
                          'kappa3_sfh': [1e-10, 1],
                          # uniform from 0 to 1. Will be tranformed to betas. 1e-10 for numerical stability.
                          'fburst': [1e-10, 1.0], 'tburst': [1e-2, 13.27],
                          'logzsol': [-2.6, 0.3],
                          'dust1': [1e-10, 3], 'dust2': [1e-10, 3], 'dust_index': [-3, 1],
                          'logm': [1e-10, 16],
                          'redshift': [1e-10, 1.5]
                          }

        self.bounds = np.array([self.prior[key] for key in self.params_name])

    def _calc_transmission(self, filterset, filter_dir=None):
        import sys
        sys.path.append('/home/jiaxuanl/Research/Packages/sedpy/')

        """
        Interploate and evaluate transmission curves at `self.wave_obs`.
        Also calculate the zeropoint in each filter.
        The interpolated transmission efficiencies are saved in `self.transmission_effiency`.
        And the zeropoint counts of each filter are saved in `self.ab_zero_counts`.

        Parameters
        ----------
        filterset: list of strings.
            Names of filters, e.g., ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'].
            You can look up at available filters using `sedpy.observate.list_available_filters`.
        """
        x = self.wavelength.cpu().detach().numpy()

        # transmission efficiency
        _epsilon = np.zeros((len(filterset), len(x)))
        _zero_counts = np.zeros(len(filterset))
        if filter_dir is None:
            filters = observate.load_filters(filterset)
        else:
            filters = observate.load_filters(filterset, directory=filter_dir)

        for i in range(len(filterset)):
            _epsilon[i] = interp1d(filters[i].wavelength,
                                   filters[i].transmission,
                                   bounds_error=False,
                                   fill_value=0)(x)
            _zero_counts[i] = filters[i].ab_zero_counts
        self.filterset = filterset
        self.filter_dir = filter_dir
        self.transmission_effiency = Tensor(_epsilon).to(self.device)
        self.ab_zero_counts = Tensor(_zero_counts).to(self.device)

    def _parse_nsa_noise_model(self, noise_model_dir):
        """
        Parse the noise model from the NSA.
        The noise model is generated in `popsed/notebook/forward_model/noise_model/``.

        Parameters
        ----------
        noise_model_dir: str. The directory of the noise model file.
        """
        meds_sigs, stds_sigs = np.load(noise_model_dir, allow_pickle=True)
        # meds_sigs is the median of noise, stds_sigs is the std of noise. All in magnitude.

        n_filters = len(meds_sigs)
        mag_grid = torch.arange(10, 30, 1)
        med_sig_grid = torch.vstack(
            [Tensor(meds_sigs[i](mag_grid)) for i in range(n_filters)])
        std_sig_grid = torch.vstack(
            [Tensor(stds_sigs[i](mag_grid)) for i in range(n_filters)])

        self.mag_grid = mag_grid.to(self.device)
        self.med_sig_grid = med_sig_grid.T.to(self.device)
        self.std_sig_grid = std_sig_grid.T.to(self.device)

    def predict(self, params):
        """
        Predict the PCA coefficients of the spectrum, given the SPS physical parameters.
        Note: this is in restframe, and the spectrum is scaled to 1 M_sun.

        Parameters
        ----------
        params (torch.Tensor): SPS physical parameters, shape = (n_samples, n_params).

        Returns
        -------
        pca_coeff (torch.Tensor): PCA coefficients, shape = (n_samples, n_pca_coeffs).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)

        return torch.hstack([_speculator.predict(params) for _speculator in self.speculators])

    def _predict_spec_restframe(self, params, log_stellar_mass=None):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.
        The predicted spectra are in restframe.

        Parameters
        ----------
        params: torch.Tensor. The SPS physical parameters, **not including stellar mass and redshift**.
            shape = (n_samples, n_params).
        log_stellar_mass: torch.Tensor or np.ndarray. log10 stellar mass of each spectrum, shape = (n_samples).

        Returns
        -------
        spec: torch.Tensor. The predicted spectra, shape = (n_wavelength, n_samples).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        if log_stellar_mass is None:
            log_stellar_mass = torch.zeros_like(params[:, 0:1])

        return torch.hstack(
            [_speculator._predict_spec_with_mass_restframe(
                torch.hstack([params, log_stellar_mass])
            ) for _speculator in self.speculators])

    def transform(self, spectra_restframe, z, islog=False):
        """
        Redshift spectra.
        Linear interpolation is used. Values outside
        the interpolation range are linearly interpolated.
        **We have to redshift the spectra after combining them in restframe.**

        Parameters
        ----------
        spectra_restframe: torch.Tensor.
            Restframe spectra in **linear** flux, shape = (n_wavelength, n_samples).
        z: torch.Tensor or np.ndarray.
            Redshifts of each spectra, shape = (n_samples,).
        islog: bool.
            Whether the input spectra is in log10-flux.
            If True, the output spectra is also in log10-flux.

        Returns
        -------
        spectra_transformed: torch.Tensor. Redshifted spectra.
        """
        if not torch.is_tensor(z):
            z = torch.tensor(z, dtype=torch.float).to(self.device)
        z = z.squeeze()

        if torch.any(z > 0):
            wave_redshifted = (self.wavelength.unsqueeze(1) * (1 + z)).T

            distances = Interp1d()(self.z_grid, self.dist_grid, z)
            # 1e5 because the absolute mag is 10pc.
            # dfactor = ((distances * 1e5)**2 / (1 + z))
            dfactor = ((distances * 1e5)**2 / (1 + z))
            # dfactor = ((distances * 1e5)**2 * (1 + z))
            # Interp1d function takes (1) the positions (`wave_redshifted`) at which you look up the value
            # in `spectrum_restframe`, learn the interpolation function, and apply it to observation wavelengths.
            if islog:
                spec = Interp1d()(wave_redshifted, spectra_restframe,
                                  self.wavelength) - torch.log10(dfactor.T)
            else:
                spec = Interp1d()(wave_redshifted, spectra_restframe, self.wavelength) / dfactor.T
            return spec
        else:
            return spectra_restframe

    def _predict_spec_with_mass_redshift(self, params,
                                         external_redshift=None):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.
        **We have to redshift the spectra after combining them in restframe.**

        Parameters
        ----------
        params: torch.Tensor.
            SPS physical parameters, including stellar mass. shape = (n_samples, n_params).
            params[:, :-2] are the SPS physical parameters NOT including stellar mass and redshift.
                If you are using non-parametric SPS model, you might include redshift (as a proxy for t_age).
            params[:, -2:-1] is the log10 stellar mass.
            params[:, -1:] is the redshift, used to shift and dim the spectra.

        Returns
        -------
        spec: torch.Tensor.
            Predicted spectra in linear scales, shape = (n_wavelength, n_samples).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        spec_rest = self._predict_spec_restframe(
            params[:, :-1], log_stellar_mass=params[:, -1:])  # restframe

        # if torch.any(torch.log10(spec_rest) > thresh):
        #     # print(f'log spec > {thresh} params:',
        #     #       params[(torch.log10(spec_rest) > thresh).any(dim=1)])
        #     print(f'log spec > {thresh} params:', (torch.log10(
        #         spec_rest) > thresh).any(dim=1).sum())
        # such that interpolation will not do linear extrapolation.
        # spec_rest[:, 0] = 0.0  # torch.nan
        # print(params[:, -2:-1])
        if external_redshift is None:
            spec = self.transform(spec_rest, params[:, -2:-1], islog=False)
        else:
            spec = self.transform(spec_rest, external_redshift, islog=False)
        spec[spec == 0.0] = 1e-30  # aviod nan = log(0)
        thresh = 15
        spec[torch.any(torch.log10(spec_rest) > thresh, dim=1)] = 1e-30
        return spec

    def predict_spec(self, params, log_stellar_mass=None, redshift=None):
        """
        Predict the corresponding spectra (in linear scale) for given physical parameters.
        The predicted spectra are in restframe.

        Parameters
        ----------
        params: torch.Tensor. The SPS physical parameters, **not including stellar mass and redshift**.
            shape = (n_samples, n_params).
        log_stellar_mass: torch.Tensor or np.ndarray. log10 stellar mass of each spectrum, shape = (n_samples).
        redshift: torch.Tensor or np.ndarray. Redshift of each spectrum, shape = (n_samples).

        Returns
        -------
        spec: torch.Tensor. The predicted spectra, shape = (n_wavelength, n_samples).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        if log_stellar_mass is None:
            log_stellar_mass = torch.zeros_like(params[:, 0:1])
        if redshift is None:
            redshift = torch.zeros_like(params[:, 0:1])

        return self._predict_spec_with_mass_redshift(torch.hstack([params, log_stellar_mass]).to(self.device),
                                                     external_redshift=redshift)

    def _predict_mag_with_mass_redshift(self, params,
                                        external_redshift=None,
                                        filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
                                        noise=None, noise_model_dir='./noise_model/nsa_noise_model_mag.npy', SNR=10):
        """
        Predict corresponding photometry (in magnitude), given SPS physical parameters.

        Parameters
        ----------
        params: torch.Tensor.
            SPS physical parameters, including stellar mass. shape = (n_samples, n_params).
            params[:, :-2] are the SPS physical parameters NOT including stellar mass and redshift.
                If you are using non-parametric SPS model, you might include redshift (as a proxy for t_age).
            params[:, -2:-1] is the log10 stellar mass.
            params[:, -1:] is the redshift, used to shift and dim the spectra.

        filterset: list.
            List of filters to predict photometry, default = ['sdss_{0}0'.format(b) for b in 'ugriz'].
        nosie: str.
            Whether to add noise to the predicted photometry.
            If `noise=None`, no noise is added.
            If `noise='nsa'`, we add noise based on NSA catalog.
            If `noise='snr'`, we add noise with constant SNR. Therefore, SNR must be provided.
        noise_model_dir: str.
            The directory of the noise model. Only works if `noise='nsa'`.
        SNR: float.
            The signal-to-noise ratio in maggies, default = 10 (results in ~0.1 mag noise in photometry).
            Only works if `noise='snr'`.

        Returns:
            mags: torch.Tensor.
                Predicted photometry, shape = (n_bands, n_samples).
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
        # lightspeed = 2.998e18  # AA/s
        # jansky_cgs = 1e-23

        # Notice: for NMF-based emulator, we train the emulator based on spectra with Lsun/Hz unit.
        # We convert Lsun/Hz to Lsun/AA, then to erg/s/AA at 10pc. The extra distance term is taken
        # into account in the `transform` (redshifting).
        if self._model == 'NMF':
            _spec = self._predict_spec_with_mass_redshift(
                params, external_redshift=external_redshift) * self.lightspeed / self.wavelength**2 * self.to_cgs_at_10pc  # / external_redshift
        else:
            raise NotImplementedError('Only NMF-based emulator is supported.')
        _spec = torch.nan_to_num(_spec, 0.0)

        maggies = torch.trapezoid(
            ((self.wavelength * _spec)[:, None, :] * self.transmission_effiency[None, :, :]
             ), self.wavelength) / self.ab_zero_counts

        maggies[maggies <= 0.] = 1e-15

        if noise == 'nsa':
            # Add noise based on NSA noise model.
            self._parse_nsa_noise_model(noise_model_dir)
            mags = -2.5 * torch.log10(maggies)  # noise-free magnitude
            _sigs_mags = torch.zeros_like(mags)
            _sig_flux = torch.zeros_like(mags)
            for i in range(maggies.shape[1]):
                _sigs_mags[:, i] = Interp1d()(self.mag_grid, self.med_sig_grid[:, i], mags[:, i])[
                    0] + Interp1d()(self.mag_grid, self.std_sig_grid[:, i], mags[:, i])[0] * torch.randn_like(mags[:, i])
                _sig_flux[:, i] = utils.sigma_mag2flux(
                    _sigs_mags[:, i], mags[:, i])  # in nanomaggies
            _noise = _sig_flux * torch.randn_like(mags) * 1e-9  # in maggies
            _noise[(maggies + _noise) < 0] = 0.0
            return -2.5 * torch.log10(maggies + _noise)

        elif noise == 'snr':
            # Add noise with constant SNR.
            _noise = torch.randn_like(maggies) * maggies / SNR
            _noise[(maggies + _noise) < 0] = 0.0
            maggies += _noise

        if torch.isnan(maggies).any() or torch.isinf(maggies).any():
            print(maggies)
        mags = -2.5 * torch.log10(maggies)

        return mags

    def _predict_mag_with_mass_redshift_batch(self, params,
                                              filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
                                              noise=None, noise_model_dir='./noise_model/nsa_noise_model_mag.npy', SNR=10):
        """
        The wrapper for `_predict_mag_with_mass_redshift` for large sample size.
        """
        size = len(params)
        if not torch.is_tensor(params):
            params = torch.tensor(params)  # still on CPU

        dataloader = DataLoader(params, batch_size=200, shuffle=False)
        mags = torch.zeros((size, len(filterset)))
        for i, data in enumerate(dataloader):
            data = data.to(self.device)
            mags[i * 200:(i + 1) * 200] = self._predict_mag_with_mass_redshift(
                data, filterset=filterset, noise=noise, noise_model_dir=noise_model_dir, SNR=SNR).clone().cpu()
        torch.cuda.empty_cache()
        return mags
        # mags = self._predict_mag_with_mass_redshift(
        #     data, filterset=filterset, noise=noise,
        #     noise_model_dir=noise_model_dir, SNR=SNR)
        # if size == 1:
        #     return mags
        # else:
        #     yield mags

    def predict_mag(self, params, log_stellar_mass=None, redshift=None, **kwargs):
        """
        Predict the corresponding magnitude for given physical parameters.

        Parameters
        ----------
        params: torch.Tensor. The SPS physical parameters, **not including stellar mass and redshift**.
            shape = (n_samples, n_params).
        log_stellar_mass: torch.Tensor or np.ndarray. log10 stellar mass of each spectrum, shape = (n_samples).
        redshift: torch.Tensor or np.ndarray. Redshift of each spectrum, shape = (n_samples).
        kwargs: you can pass filterset and noise model here. See `self._predict_mag_with_mass_redshift`.

        Returns
        -------
        mag: torch.Tensor. The predicted magnitudes, shape = (n_wavelength, n_samples).
        """
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        if log_stellar_mass is None:
            log_stellar_mass = torch.zeros_like(params[:, 0:1])
        if redshift is None:
            redshift = torch.zeros_like(params[:, 0:1])
        return self._predict_mag_with_mass_redshift(torch.hstack([params, log_stellar_mass]).to(self.device), external_redshift=redshift,
                                                    **kwargs)
