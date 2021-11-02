'''
Modified Speculator, based on https://github.com/justinalsing/speculator/blob/master/speculator/speculator.py
'''

import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


class SpectrumPCA():
    """
    SPECULATOR PCA compression class
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
        nn.BatchNorm1d(output_size),
        CustomActivation(output_size),
        nn.Dropout(p=0.5)
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
    def __init__(self, n_parameters=None, n_pca_components=None, wavelengths=None, pca_filename=None, hidden_size=None) -> None:
        self.n_parameters = n_parameters  # e.g., n_parameters = 9
        self.hidden_size = hidden_size  # e.g., [100, 100, 100]
        self.n_pca_components = n_pca_components  # e.g., n_pca_components = 20
        # e.g., wavelengths = np.arange(3800, 7000, 2)
        # self.wavelengths = wavelengths
        self.pca_filename = pca_filename
        with open(self.pca_filename, 'rb') as f:
            self.pca = pickle.load(f)
        # normalization of log spectrum
        self.log_spectrum_scale = self.pca.log_spectrum_scale
        # normalization of log spectrum
        self.log_spectrum_shift = self.pca.log_spectrum_shift

        self.network = Network(
            self.n_parameters, self.hidden_size, self.n_pca_components)

        self.train_loss_history = []
        self.val_loss_history = []

    def load_data(self, pca_coeff, params, val_frac=0.2, batch_size=32):
        # Normalize PCA coefficients
        self.pca_shift = np.median(pca_coeff, axis=0)
        self.pca_scale = np.std(pca_coeff, axis=0)
        self.param_shift = np.median(params, axis=0)
        self.param_scale = np.std(params, axis=0)
        # self.pca_scale[0] = 2

        # pca_coeff = (pca_coeff - self.pca_shift) / self.pca_scale
        # params = (params - self.param_shift) / self.param_scale

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
        dataloaders['val'] = DataLoader(val_data, batch_size=val_len)

        self.dataloaders = dataloaders

    def train(self, learning_rate=0.002, n_epochs=50, display=False):
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
                    inputs = Variable(inputs)
                    labels = Variable(labels)  # .double()

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

                if epoch % 10 == 0:
                    print(
                        'Epoch: {} - {} Loss: {:.4f}'.format(epoch, phase, epoch_loss))

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
        params = torch.FloatTensor(params)
        pca_coeff = self.network(params).detach().numpy()
        pca_coeff = pca_coeff * self.pca_scale + self.pca_shift

        return pca_coeff

    def predict_spec(self, params):
        params = torch.FloatTensor(params)
        pca_coeff = self.network(params).detach().numpy()
        pca_coeff = pca_coeff * self.pca_scale + self.pca_shift

        spec = self.pca.PCA.inverse_transform(
            pca_coeff) * self.log_spectrum_scale + self.log_spectrum_shift

        return spec

    def save_model(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load_model(self, filename):
        self.network.load_state_dict(torch.load(filename))
