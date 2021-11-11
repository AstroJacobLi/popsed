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

from sedpy import observate
# from sklearn.preprocessing import StandardScaler


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
        else: # numpy array
            dims = list(range(values.ndim - 1))
            self.mean = np.mean(values, axis=tuple(dims))
            self.std = np.std(values, axis=tuple(dims))

    def transform(self, values, device=None):
        if device is None:
            device = self.device

        if torch.is_tensor(values):
            return (values - Tensor(self.mean).to(device)) / Tensor(self.std + self.epsilon).to(device)
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
    Implementation of the activation function described in `speculator` paper

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
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        # nn.BatchNorm1d(output_size),
        CustomActivation(output_size),
        # nn.Dropout(p=0.5)
    )


class Network(nn.Module):
    '''
    Fully connected network with dropout and batch normalization
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
    def __init__(self, n_parameters=None, n_pca_components=None, pca_filename=None, hidden_size=None) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.n_parameters = n_parameters  # e.g., n_parameters = 9
        self.hidden_size = hidden_size  # e.g., [100, 100, 100]
        # e.g., wavelengths = np.arange(3800, 7000, 2)
        # self.wavelengths = wavelengths
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

    def load_data(self, pca_coeff, params, val_frac=0.2, batch_size=32):
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

    def train(self, learning_rate=0.002, n_epochs=50, display=False):
        the_last_loss = 1e-3
        min_loss = 1e-2
        min_recon_err = 1e-3
        patience = 20
        trigger_times = 0

        optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        # Here I use a weighted MSE loss to emphasize the first element of PCA coefficients
        # def weighted_mse_loss(input, target):
        #     weight = torch.ones(self.n_pca_components)
        #     weight[0] = 3
        #     return torch.sum(weight * (input - target) ** 2)
        # loss_fn = weighted_mse_loss

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
                        # outputs = torch.add(torch.mul(outputs.double(), torch.from_numpy(
                        #     self.pca_scale).double()), torch.from_numpy(self.pca_shift).double())
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
                self.save_model('best_loss_model.pkl')
                self.best_loss_epoch = epoch

            for x, y in self.dataloaders['val']:
                spec = self.predict_spec(x)
                spec_y = self.predict_spec_from_norm_pca(y)
            
            recon_err = torch.median(torch.abs((spec - spec_y) / torch.abs(spec_y)))
            if recon_err < min_recon_err:
                min_recon_err = recon_err
                self.save_model('best_recon_err_model.pkl')
                self.best_recon_err_epoch = epoch

        if display:
            self.plot_loss()

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        plt.plot(np.array(self.val_loss_history).flatten(), label='Val loss')
        plt.legend()

        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def predict(self, params):
        if not torch.is_tensor(params):
            params = torch.Tensor(params).to(self.device)
        params = params.to(self.device)
        self.network.eval()
        pca_coeff = self.network(params)
        pca_coeff = self.pca_scaler.inverse_transform(pca_coeff, device=self.device)

        return pca_coeff

    def predict_spec(self, params):
        pca_coeff = self.predict(params)
        spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device)

        return spec

    def predict_spec_from_norm_pca(self, y):
        pca_coeff = self.pca_scaler.inverse_transform(y.to(self.device), device=self.device)
        spec = self.pca.logspec_scaler.inverse_transform(self.pca.inverse_transform(
            pca_coeff, device=self.device), device=self.device)

        return spec

    def predict_sed(self, params, filterset=['sdss_{0}0'.format(b) for b in 'ugriz'], angstroms=np.arange(3000, 11000, 2)):
        '''
        Predict magnitudes for a given set of filters.
        See https://github.com/bd-j/prospector/blob/dda730feef5b8e679864521d0ac1c5f5f3db989c/prospect/models/sedmodel.py#L591
        '''
        # get magnitude from a spectrum
        lightspeed = 2.998e18  # AA/s
        jansky_cgs = 1e-23
        
        f_maggies = 10**self.predict_spec(params).cpu().detach().numpy()
        f_lambda_cgs = f_maggies * lightspeed / angstroms**2 * (3631 * jansky_cgs)
        filterlist = observate.load_filters(filterset)
        mags = observate.getSED(angstroms, f_lambda_cgs, filterlist=filterlist)
        return mags


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        #torch.save(self, filename)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)