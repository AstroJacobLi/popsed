'''
Neural density estimator for population-level inference. 
'''
import gc
import torch
from torch import nn, Tensor, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nflows import flows, transforms
from nflows.nn import nets
from nflows import distributions as distributions_

from sbi.utils.sbiutils import standardizing_transform
from sbi.utils.torchutils import create_alternating_binary_mask

import copy
import os
from tqdm import trange
import dill as pickle
import numpy as np

from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from popsed.speculator import StandardScaler
from geomloss import SamplesLoss


"""
I steal the NF code from https://github.com/mackelab/sbi/blob/019fde2d61edbf8b4a02e034dc9c056b0d240a5c/sbi/neural_nets/flow.py#L77
But here everything is NOT conditioned.
"""


def build_maf(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    device: str = 'cuda',
    initial_pos: dict = {'bounds': [[1, 2], [0, 1]], 'std': [1, .05]},
    **kwargs,
):
    """Builds MAF to describe p(x).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network, i.e., whether do normalization.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()

    if x_numel == 1:
        raise Warning(
            f"In one-dimensional output space, this flow is limited to Gaussians")

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=x_numel,
                        hidden_features=hidden_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=torch.tanh,
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=x_numel),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    if z_score_x:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if initial_pos is not None:
        _mean = np.random.uniform(
            low=np.array(initial_pos['bounds'])[:, 0], high=np.array(initial_pos['bounds'])[:, 1])
        print(_mean)
        transform_init = transforms.AffineTransform(shift=torch.Tensor(-_mean) / torch.Tensor(initial_pos['std']),
                                                    scale=1.0 / torch.Tensor(initial_pos['std']))
        transform = transforms.CompositeTransform([transform_init, transform])

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net).to(device)

    return neural_net


def build_nsf(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    device: str = 'cuda',
    initial_pos: dict = {'bounds': [[1, 2], [0, 1]], 'std': [1, .05]},
    **kwargs,
):
    """Builds NSF to describe p(x).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        initial_pos: only works when z_score_x is True.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
    Returns:
        Neural network.
    """
    x_numel = batch_x[0].numel()

    if x_numel == 1:

        class ContextSplineMap(nn.Module):
            """
            Neural network from `context` to the spline parameters.
            We cannot use the resnet as conditioner to learn each dimension conditioned
            on the other dimensions (because there is only one). Instead, we learn the
            spline parameters directly. In the case of conditinal density estimation,
            we make the spline parameters conditional on the context. This is
            implemented in this class.
            """

            def __init__(
                self,
                in_features: int,
                out_features: int,
                hidden_features: int,
                context_features: int,
            ):
                """
                Initialize neural network that learns to predict spline parameters.
                Args:
                    in_features: Unused since there is no `conditioner` in 1D.
                    out_features: Number of spline parameters.
                    hidden_features: Number of hidden units.
                    context_features: Number of context features.
                """
                super().__init__()
                # `self.hidden_features` is only defined such that nflows can infer
                # a scaling factor for initializations.
                self.hidden_features = hidden_features

                # Use a non-linearity because otherwise, there will be a linear
                # mapping from context features onto distribution parameters.
                self.spline_predictor = nn.Sequential(
                    nn.Linear(context_features, self.hidden_features),
                    nn.ReLU(),
                    nn.Linear(self.hidden_features, self.hidden_features),
                    nn.ReLU(),
                    nn.Linear(self.hidden_features, out_features),
                )

            def __call__(
                self, inputs: Tensor, context: Tensor, *args, **kwargs
            ) -> Tensor:
                """
                Return parameters of the spline given the context.
                Args:
                    inputs: Unused. It would usually be the other dimensions, but in
                        1D, there are no other dimensions.
                    context: Context features.
                Returns:
                    Spline parameters.
                """
                return self.spline_predictor(context)

        def mask_in_layer(i): return tensor([1], dtype=uint8)

        def conditioner(in_features, out_features): return ContextSplineMap(
            in_features, out_features, hidden_features, context_features=None
        )
        if num_transforms > 1:
            raise Warning(
                f"You are using `num_transforms={num_transforms}`. When estimating a "
                f"1D density, you will not get any performance increase by using "
                f"multiple transforms with NSF. We recommend setting "
                f"`num_transforms=1` for faster training (see also 'Change "
                f"hyperparameters of density esitmators' here: "
                f"https://www.mackelab.org/sbi/tutorial/04_density_estimators/)."
            )

    else:
        def mask_in_layer(i): return create_alternating_binary_mask(
            features=x_numel, even=(i % 2 == 0)
        )

        def conditioner(in_features, out_features): return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=2,
            activation=nn.ReLU(),
            dropout_probability=0.0,
            use_batch_norm=False,
        )

    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.PiecewiseRationalQuadraticCouplingTransform(
                        mask=mask_in_layer(i),
                        transform_net_create_fn=conditioner,
                        num_bins=num_bins,
                        tails="linear",
                        tail_bound=3.0,
                        apply_unconditional_transform=False,
                    ),
                    transforms.LULinear(x_numel, identity_init=True),
                ]
            )
            for i in range(num_transforms)
        ]
    )

    if z_score_x:
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])

    if initial_pos is not None:
        _mean = np.random.uniform(
            low=np.array(initial_pos['bounds'])[:, 0], high=np.array(initial_pos['bounds'])[:, 1])
        print(_mean)
        transform_init = transforms.AffineTransform(shift=torch.Tensor(-_mean) / torch.Tensor(initial_pos['std']),
                                                    scale=1.0 / torch.Tensor(initial_pos['std']))
        transform = transforms.CompositeTransform([transform_init, transform])

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net).to(device)

    return neural_net, _mean


class NeuralDensityEstimator(object):
    """
    Neural density estimator class. Basically a wrapper.
    """

    def __init__(
            self,
            normalize: bool = True,
            initial_pos: dict = None,
            method: str = "nsf",
            hidden_features: int = 50,
            num_transforms: int = 5,
            num_bins: int = 10,
            embedding_net: nn.Module = nn.Identity(),
            **kwargs):
        """
        Initialize neural density estimator.

        Parameters
        ----------
        normalize: Whether to z-score the data that you want to model.
        initial_pos: Initial position of the density, 
            e.g., `{'bounds': [[1, 2], [0, 1]], 'std': [1, .05]}`.
            It includes the bounds for sampling the means of Gaussians, 
            and the standard deviations of the Gaussians.
        method: Method to use for density estimation, either 'nsf' or 'maf'.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        assert method in [
            'nsf', 'maf'], "Method must be either 'nsf' or 'maf'."
        self.method = method

        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_bins = num_bins  # only works for NSF
        self.normalize = normalize

        if initial_pos is None:
            raise ValueError(
                "initial_pos must be specified. Please see the documentation.")
        assert len(initial_pos['bounds']) == len(
            initial_pos['std']), "The length of bounds and std must be the same."
        self.initial_pos = initial_pos

        self.embedding_net = embedding_net
        self.train_loss_history = []
        self.vali_loss_history = []

    def build(self, batch_theta: Tensor, optimizer: str = "adam",
              lr=1e-3, **kwargs):
        """
        Build the neural density estimator based on input data.

        Parameters
        ----------
        batch_theta (torch.Tensor): the input data whose distribution will be modeled by NDE.
        optimizer (float): the optimizer to use for training, default is Adam.
        lr (float): learning rate for the optimizer.

        """
        if not torch.is_tensor(batch_theta):
            batch_theta = torch.tensor(batch_theta, device=self.device)
        self.batch_theta = batch_theta

        if self.method == "maf":
            self.net = build_maf(
                batch_x=batch_theta,
                z_score_x=self.normalize,
                initial_pos=self.initial_pos,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                embedding_net=self.embedding_net,
                device=self.device,
                **kwargs
            )
        elif self.method == "nsf":
            self.net, self.mean_init = build_nsf(
                batch_x=batch_theta,
                z_score_x=self.normalize,
                initial_pos=self.initial_pos,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                num_bins=self.num_bins,
                embedding_net=self.embedding_net,
                device=self.device,
                **kwargs
            )

        self.net.to(self.device)

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer}, only support 'Adam' now.")

    def _train(self, n_epochs: int = 2000, display=False, suffix: str = "nde"):
        """
        Train the neural density estimator based on input data.
        Here we use the log(P) loss. This function is not used in the project.

        Parameters
        ----------
        n_epochs: Number of epochs to train.
        display: Whether to display the training loss.
        suffix: Suffix to add to the output file.

        """
        min_loss = -19
        patience = 5
        self.best_loss_epoch = 0
        self.net.train()

        for epoch in trange(n_epochs, desc='Training NDE', unit='epochs'):
            self.optimizer.zero_grad()
            loss = -self.net.log_prob(self.batch_x).mean()
            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())

            if loss.item() < min_loss:
                min_loss = loss.item()
                if epoch - self.best_loss_epoch > patience:
                    # Don't save model too frequently
                    self.best_loss_epoch = epoch
                    self.save_model(
                        f'best_loss_model_{suffix}_{self.method}.pkl')

        if min_loss == -18:
            raise Warning('The training might be failed, try more epochs')

        if display:
            self.plot_loss()

    def sample(self, n_samples: int = 1000):
        """
        Sample according to the fitted NDE

        Parameters
        ----------
        n_samples: Number of samples to draw.

        Returns
        -------
        samples: Samples drawn from the NDE.
        """
        return self.net.sample(n_samples)

    def plot_loss(self, min_loss=0.017):
        """
        Display the loss curves.

        Parameters
        ----------
        min_loss: a horizontal line at `min_loss` will be shown.

        """
        # min_loss is the intrinsic minimum loss
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        if hasattr(self, 'vali_loss_history'):
            plt.plot(np.array(self.vali_loss_history).flatten(),
                     label='Validation loss')
        plt.axhline(y=min_loss, color='r', linestyle='--',
                    label='Intrinsic minimum')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    def save_model(self, filename):
        """
        Save NDE model.

        Parameters
        ----------
        filename: Name of the file to save the model.

        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


class WassersteinNeuralDensityEstimator(NeuralDensityEstimator):
    """
    Wasserstein Neural Density Estimator. We use this class in the paper.
    """

    def __init__(
            self,
            normalize: bool = True,
            initial_pos: dict = None,
            method: str = "nsf",
            sps_model: str = 'NMF',
            seed: int = None,
            hidden_features: int = 50,
            num_transforms: int = 5,
            num_bins: int = 10,
            embedding_net: nn.Module = nn.Identity(),
            output_dir='./nde_theta/',
            regularize=False,
            NDE_prior=None,
            **kwargs):
        """
        Initialize Wasserstein Neural Density Estimator.

        Parameters
        ----------
        normalize: bool, whether to normalize the input data. Default is True.
        initial_pos: dict, the initial position of the Gaussians in NF. 
            E.g., `{'bounds': [[1, 2], [0, 1]], 'std': [1, .05]}`.
            It includes the bounds for sampling the means of Gaussians, 
            and the standard deviations of the Gaussians.
        method: str, the method to use for NDE. Default is 'nsf'. 
            Only support 'nsf' and 'maf' now.
        sps_model: str, the model to use for SPS. Default is 'NMF'.
        seed: int, random seed. If None, will randomly generate one.
        hidden_features: int, number of hidden features.
        num_transforms: int, number of transforms.
        num_bins: int, number of bins. Only works for `method='nsf'`.
        embedding_net: nn.Module, the embedding net. Default is nn.Identity().
        output_dir: str, the output directory. Default is './nde_theta/'.
        regularize: bool, whether to transform the physical parameters using Gaussian CDF. 
            Default is False.
        NDE_prior: array, the prior (tophat bounds) used to do the transformation.
        """
        super(WassersteinNeuralDensityEstimator, self).__init__(
            normalize=normalize,
            initial_pos=initial_pos,
            method=method,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_bins=num_bins,
            embedding_net=embedding_net,
            **kwargs
        )
        assert sps_model in ['NMF', 'tau'], 'Only support `NMF` and `tau` now.'
        self.sps_model = sps_model
        self.initial_pos = initial_pos
        self.patience = 1
        self.min_loss = 0.2
        self.best_loss_epoch = 0

        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 1000)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.output_dir = output_dir
        if (self.output_dir is not None):
            try:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
            except:
                pass

        # Set penalty term
        if self.sps_model == 'NMF':
            # self.penalty_powers = [10] * 11
            # self.penalty_powers = [30] * 3 + [30] * 2 + [30] + [20] * 3 + [30] * 2 # 30 for dust1 and dust2
            self.penalty_powers = [50] * 3 + [50] * 6 + [50] + [100] + [50]
            #[100] * 9 + [500] + [100] + [500]
        else:
            self.penalty_powers = [100, 100, 100, 100, 50, 500]

        # Regularize parameters, i.e., transform them with Gaussian CDF
        if regularize and NDE_prior is None:
            raise ValueError(
                'NDE_prior must be provided when regularize is True.')
        self.regularize = regularize
        self.NDE_prior = NDE_prior

    def build(self,
              batch_theta: Tensor,
              batch_X: Tensor,
              z_score=True,
              filterset: list = ['sdss_{0}0'.format(b) for b in 'ugriz'],
              optimizer: str = "adam",
              lr=1e-3, **kwargs):
        """
        Build the neural density estimator based on input data.

        Parameters
        ----------
        batch_theta: Tensor, the stellar population parameters input parameters.
            This is only needed to construct the NDE (i.e., need dimensions for network).
        batch_X: Tensor, the input photometry data.
        filterset: list, the filterset to use. Default is ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'].
        optimizer: str, the optimizer to use. Default is 'adam'.
        lr: float, the learning rate. Default is 1e-3.

        """
        from torch.utils.data import DataLoader
        super().build(batch_theta, optimizer, lr, **kwargs)
        
        if not torch.is_tensor(batch_X):
            batch_X = torch.tensor(batch_X, device='cpu')

        # We z-score the input photometry data
        # scaler = StandardScaler(device=self.device)
        scaler = StandardScaler(device='cpu')
        # because batch_X is on CPU, we have to first train on CPU
        scaler.fit(batch_X)
        self.scaler = scaler
        self.z_score = z_score
        # dataloader = DataLoader(batch_X, batch_size=2000, shuffle=True)
        # self.X = [yield scaler.transform(x, device='cpu').detach() for x in dataloader]
        if self.z_score:
            self.X = self.scaler.transform(
                batch_X).detach()  # z-scored observed SEDs
        else:
            self.X = batch_X.detach()

        self.X = self.X.to(self.device)

        self.filterset = filterset
        # scaler.device = self.device

    def load_validation_data(self, X_vali):
        """
        Load validation data.

        Parameters
        ----------
        X_vali: Tensor, the photometry for validation.
        """
        if not torch.is_tensor(X_vali):
            X_vali = torch.tensor(X_vali, device='cpu')
        if self.z_score:
            self.X_vali = self.scaler.transform(
                X_vali, device=X_vali.device).detach()  # z-scored observed SEDs
        else:
            self.X_vali = X_vali.detach()

        self.X_vali = self.X_vali.to(self.device)

    def _get_loss(self, X, speculator, n_samples,
                  noise, SNR, noise_model_dir,
                  loss_fn):
        sample = self.sample(n_samples)
        Y = self.scaler.transform(
            speculator._predict_mag_with_mass_redshift(
                sample, filterset=self.filterset,
                noise=noise, SNR=SNR, noise_model_dir=noise_model_dir)
        )
        bad_mask = torch.stack([((sample < self.bounds[i][0]) | (sample > self.bounds[i][1]))[
            :, i] for i in range(len(self.bounds))]).sum(dim=0, dtype=bool)
        Y = Y[~bad_mask]
        # Y = self.scaler.transform(
        #     speculator._predict_mag_with_mass_redshift(
        #         self.sample(n_samples), filterset=self.filterset,
        #         noise=noise, SNR=SNR, noise_model_dir=noise_model_dir)
        # )
        # bad_mask = (torch.isnan(Y).any(dim=1) | torch.isinf(Y).any(dim=1))
        # print('Bad mask num', bad_mask.sum())
        # bad_ratio = bad_mask.sum() / len(Y)
        # Y = Y[~bad_mask]
        # val = 10.0
        # Y = torch.nan_to_num(Y, val, posinf=val, neginf=-val)
        # - torch.log10(1 - bad_ratio) / 10
        powers = torch.Tensor(self.penalty_powers).to(self.device)
        penalty = log_prior(sample,
                            torch.Tensor(speculator.bounds).to(self.device), powers)
        penalty = penalty[~torch.isinf(penalty)].mean()
        loss = loss_fn(X, Y) + penalty
        # print(penalty)
        # loss = loss_fn(X, Y) - 10 * torch.log10(1 - bad_ratio) # bad_ratio * 5
        # loss = torch.log10(loss_fn(X, Y)) + torch.log10(loss_fn(X[:, 1:4], Y[:, 1:4])) - torch.log10(1 - bad_ratio)
        #+ torch.exp(5 * bad_ratio)

        return loss, penalty

    def _get_loss_NMF(self, X, speculator, n_samples,
                      noise, SNR, noise_model_dir,
                      loss_fn, add_penalty=False, regularize=False):
        """
        The most important funcgtion in this class. This defines the loss.
        This function only works for NMF-based SPS.

        Parameters
        ----------
        X: Tensor, the observed photometry data (after being z-scored). 
            Typically, X = self.X.
        speculator: SuperSpeculator, which is the emulator for the SEDs.
        n_samples: int, the number of samples to use. Recommended to be ~5000.
        noise: float, the noise model. Either 'snr' or 'nsa' or None. 
            If 'snr', you also need to provide the SNR argument.
            If 'nsa', you also need to provide the noise_model_dir argument.
        SNR: float, the signal-to-noise ratio to use if noise is 'snr'.
        noise_model_dir: str, the directory of the NSA noise model.
        loss_fn: the loss function to use. Here we use Wasserstein loss.
        add_penalty: bool, whether to only add the penalty term to loss.
        regularize: bool. Whether the SED params are transformed using log10 and sigmoid.

        Returns
        -------
        loss: Tensor, the loss.
        penalty: Tensor, the penalty term.
        """
        assert noise in [None, 'snr', 'gama',
                         'nsa'], 'Only support `snr`, `nsa`, `gama`, or `None` now.'

        if regularize:
            if hasattr(self, "cdf_z") and self.cdf_z is not None:
                sample = inverse_transform_nmf_params_given_z(
                    self.sample(n_samples), self.NDE_prior, self.cdf_z)
            elif hasattr(self, "cdf_mass") and self.cdf_mass is not None:
                sample = inverse_transform_nmf_params_given_mass(
                    self.sample(n_samples), self.NDE_prior, self.cdf_mass)
            else:
                sample = inverse_transform_nmf_params(
                    self.sample(n_samples), self.NDE_prior)
        else:
            sample = self.sample(n_samples)

        if hasattr(self, 'anpe_mass_given_z') and self.anpe_mass_given_z is not None:
            # with torch.no_grad():
            sample = torch.hstack(
                [sample, self.anpe_mass_given_z.sample(1, context=sample[:, -1:])[:, 0]])

        if hasattr(self, 'anpe_mass_given_all') and self.anpe_mass_given_all is not None:
            # with torch.no_grad():
            sample = torch.hstack(
                [sample, self.anpe_mass_given_all.sample(1, context=sample[:, :])[:, 0]])

        # if self.external_redshift_data is not None:
        #     _z = torch.Tensor(np.random.choice(self.external_redshift_data, n_samples)[
        #         :, None]).to(self.device)
        #     sample = torch.hstack([sample[:, :-1], _z, sample[:, -1:]])

        # penalty term
        # powers = torch.Tensor(self.penalty_powers).to(self.device)
        # penalty = log_prior(sample,
        #                     torch.Tensor(self.NDE_prior).to(self.device),
        #                     powers)
        # # print('Number of inf:', torch.isinf(penalty).sum())
        # penalty = penalty[~torch.isinf(penalty)].nanmean()
        
        
        Y = speculator._predict_mag_with_mass_redshift(
                sample, filterset=self.filterset,
                noise=noise, SNR=SNR, noise_model_dir=noise_model_dir)
        if self.z_score:
            Y = self.scaler.transform(
                mags,
                device=self.device
            )
        bad_mask = torch.stack([((sample < self.bounds[i][0]) | (sample > self.bounds[i][1]))[
            :, i] for i in range(len(self.bounds))]).sum(dim=0, dtype=bool)
        bad_mask |= (torch.isnan(Y).any(axis=1) |
                        torch.isinf(Y).any(axis=1))
        Y = Y[~bad_mask]
        # print('Bad mask num', bad_mask.sum())
        # penalty = torch.sum(Y[:, 2] > (19.65 - self.scaler.mean[2]) / self.scaler.std[2]) / len(Y) * 10
        penalty = torch.sum(Y[:, 2] > 19.65) / len(Y) * 10
        # dataloader = DataLoader(X, batch_size=n_samples, shuffle=True)
        # data_loss = 0.
        # for x in dataloader:
        #     data_loss += loss_fn(Y, x.to(self.device))
        
        # loss = data_loss / len(dataloader)  # + penalty
        
        loss = loss_fn(Y, X)

        if add_penalty:
            # loss += loss_fn((1 * (X[:100, 2:3].clone() - 19.65)), (1 * (Y[:100, 2:3].clone() - 19.65))) 
            loss += loss_fn(10**(1 * (X[:, 2:3].clone() - 19.65)), 10**(1 * (Y[:, 2:3].clone() - 19.65)))

        sample = None
        x = None
        y = None
        Y = None
        dataloader = None
        X = X.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        # Y.to('cpu')
        # loss_fn(X, Y)# penalty +

        return loss, penalty #torch.zeros_like(loss)  # penalty

    def train(self,
              n_epochs: int = 100,
              lr=1e-3,
              speculator=None,
              noise='nsa',
              SNR=20,
              noise_model_dir=None,
              sinkhorn_kwargs={'p': 1, 'blur': 0.01, 'scaling': 0.8},
              scheduler=None,
              add_penalty=False,
              detect_anomaly=False):
        """
        Train the neural density estimator using Wasserstein loss.

        Parameters
        ----------
        n_epochs: int, the number of epochs to train.
        lr: float, the learning rate.
        speculator: SuperSpeculator, which is the emulator for the SEDs.
        noise: str, the noise model. Either 'snr' or 'nsa' or None.
            If 'snr', you also need to provide the SNR argument.
            If 'nsa', you also need to provide the noise_model_dir argument.
        SNR: float, the signal-to-noise ratio to use if noise is 'snr'.
        noise_model_dir: str, the directory of the NSA noise model.
        sinkhorn_kwargs: dict, the kwargs for the sinkhorn loss. 
            See https://www.kernel-operations.io/geomloss/api/pytorch-api.html
        scheduler: torch.optim.lr_scheduler, the learning rate scheduler.
        only_penalty: bool, whether to only use the penalty term as loss.
        regularize: bool. Whether the SED params are transformed using log10 and sigmoid.
        detect_anomaly: bool, whether to detect the anomaly.
        """
        torch.autograd.set_detect_anomaly(detect_anomaly)

        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        L = SamplesLoss(loss="sinkhorn", **sinkhorn_kwargs)

        # learning rate
        if scheduler is not None:
            self.optimizer.param_groups[0]['lr'] = lr

        t = trange(n_epochs,
                   desc='Training NDE_theta using Wasserstein loss',
                   unit='epochs')

        for epoch in t:
            self.optimizer.zero_grad()
            X_train, _ = train_test_split(
                self.X.detach(), test_size=0.2, shuffle=True)
            # n_samples = len(X_train)
            n_samples = 8000
            # aggr_loss = 0
            # for i in range(3):
            loss, bad_ratio = self._get_loss_NMF(X_train, speculator, n_samples,
                                                 noise, SNR, noise_model_dir, L,
                                                 add_penalty=add_penalty, regularize=self.regularize)
            #     aggr_loss += loss
            # aggr_loss /= 3
            # aggr_loss.backward()
            # t.set_description(
            #     f'Loss = {loss.item():.3f} (train), {bad_ratio.item():.3f} (bad ratio)')
            loss.backward()
            self.optimizer.step()
            self.train_loss_history.append(loss.item())

            # get validation loss
            vali_loss, _ = self._get_loss_NMF(self.X_vali, speculator, n_samples,  # len(self.X_vali),
                                              noise, SNR, noise_model_dir, L,
                                              add_penalty, regularize=self.regularize)
            self.vali_loss_history.append(vali_loss.item())

            t.set_description(
                f'Loss = {loss.item():.3f} (train), {vali_loss.item():.3f} (vali), {bad_ratio.item():.3f} (bad ratio)')

            if torch.isnan(loss):
                print('Stop training because the loss is NaN.')
                break

            # Save the model if the loss is the best so far
            if vali_loss.item() < self.min_loss:
                self.min_loss = vali_loss.item()
                self.best_loss_epoch = len(self.vali_loss_history)
                self.best_model_state = self.net.state_dict()
                if self.output_dir is not None:
                    self.save_model(
                        os.path.join(self.output_dir,
                                     f'nde_theta_best_loss_{self.method}_{self.seed}.pkl')
                    )
            if scheduler is not None:
                scheduler.step()

    def goodness_of_fit(self, Y_truth, p=2):
        """
        Compare the recovered P(theta|D) with the ground truth. 
        Only works for mock observations (since you know the ground truth).

        Parameters
        ----------
        Y_truth: Tensor, the ground truth SED parameters.
        p: int, the p-norm to use for sinkhorn loss.

        Returns
        -------
        float, the log10 of sinkhorn loss, which represents the goodness of fit.
        """
        samples = self.sample(len(Y_truth))
        # very close to Wasserstein loss
        L = SamplesLoss(loss="sinkhorn", p=p, blur=0.001, scaling=0.95)
        self._goodness_of_fit = np.log10(
            L(Y_truth, samples).item())  # The distance in theta space
        print('Log10 Wasserstein distance in theta space: ', self._goodness_of_fit)
        return self._goodness_of_fit


def _diff_KL_w2009_eq29(X, Y, silent=True, p=1):
    """
    This function is not accurate! Just a rough estimation when X and Y are very different! 
    PyTorch/Faiss implementation of the KL divergence from Wang 2009 eq. 29.

    kNN KL divergence estimate using Eq. 29 from Wang et al. (2009). 
    This has some bias reduction applied to it and a correction for 
    epsilon.

    Sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    """
    # import faiss
    # import faiss.contrib.torch_utils

    if not torch.is_tensor(X):
        raise ValueError('The input X must be tensor.')
    if not torch.is_tensor(Y):
        raise ValueError('The input Y must be tensor.')

    assert X.shape[1] == Y.shape[1]
    n, d = X.shape  # X sample size, dimensions
    m = Y.shape[0]  # Y sample size

    # first determine epsilon(i)
    dNN1_XX = torch.cdist(X, X, p=p).sort()
    dNN1_XY = torch.cdist(X, Y, p=p).sort()

    eps = torch.amax(torch.cat(
        (dNN1_XX.values[:, 1:2], dNN1_XY.values[:, 0:1]), dim=1), 1) * 1.000001

    if not silent:
        print('  epsilons ', eps)

    # find l_i and k_i, fast now
    res = eps[:, None] - dNN1_XX.values
    mask = F.threshold(-F.threshold(res, 0., 0.), -1e-9, 1)
    l_i = torch.sum(mask, dim=1) - 1
    rho_i = torch.amax(torch.mul(dNN1_XX.values, mask), dim=1)

    res = eps[:, None] - dNN1_XY.values
    mask = F.threshold(-F.threshold(res, 0., 0.), -1e-9, 1)
    k_i = torch.sum(mask, dim=1)
    nu_i = torch.amax(torch.mul(dNN1_XY.values, mask), dim=1)

    if not silent:
        print('  l_i ', l_i)
        print('  k_i ', k_i)

    assert rho_i.min() >= 0., 'duplicate elements in your chain'

    # mask = ~torch.isinf(torch.log(rho_i / nu_i))
    d_corr = -d / n * torch.nansum(torch.log(rho_i / nu_i))

    if not silent:
        print('  first term = %f' % d_corr)
    digamma_term = torch.sum(torch.digamma(l_i) - torch.digamma(k_i)) / n
    if not silent:
        print('  digamma term = %f' % digamma_term)

    # print('   KL =', d_corr + digamma_term + np.log(float(m)/float(n-1)))
    # l_i, k_i, rho_i, nu_i
    return d_corr + digamma_term + np.log(float(m) / float(n - 1))


def _KL_w2009_eq29(X, Y, silent=True):
    ''' kNN KL divergence estimate using Eq. 29 from Wang et al. (2009). 
    This has some bias reduction applied to it and a correction for 
    epsilon.
    sources 
    ------- 
    - Q. Wang, S. Kulkarni, & S. Verdu (2009). Divergence Estimation for Multidimensional Densities Via k-Nearest-Neighbor Distances. IEEE Transactions on Information Theory, 55(5), 2392-2405.
    '''
    if torch.is_tensor(X):
        X = X.cpu().detach().numpy()
    if torch.is_tensor(Y):
        Y = Y.cpu().detach().numpy()

    assert X.shape[1] == Y.shape[1]
    n, d = X.shape  # X sample size, dimensions
    m = Y.shape[0]  # Y sample size

    # first determine epsilon(i)
    NN_X = NearestNeighbors(n_neighbors=1).fit(X)
    NN_Y = NearestNeighbors(n_neighbors=1).fit(Y)
    dNN1_XX, _ = NN_X.kneighbors(X, n_neighbors=2)
    dNN1_XY, _ = NN_Y.kneighbors(X)
    eps = np.amax([dNN1_XX[:, 1], dNN1_XY[:, 0]], axis=0) * 1.000001
    if not silent:
        print('  epsilons ', eps)

    # find l_i and k_i
    _, i_l = NN_X.radius_neighbors(X, eps)
    _, i_k = NN_Y.radius_neighbors(X, eps)
    l_i = np.array([len(il) - 1 for il in i_l])
    k_i = np.array([len(ik) for ik in i_k])
    assert l_i.min() > 0
    assert k_i.min() > 0
    if not silent:
        print('  l_i ', l_i)
        print('  k_i ', k_i)

    rho_i = np.empty(n, dtype=float)
    nu_i = np.empty(n, dtype=float)
    for i in range(n):
        rho_ii, _ = NN_X.kneighbors(
            np.atleast_2d(X[i]), n_neighbors=l_i[i] + 1)
        nu_ii, _ = NN_Y.kneighbors(np.atleast_2d(X[i]), n_neighbors=k_i[i])
        rho_i[i] = rho_ii[0][-1]
        nu_i[i] = nu_ii[0][-1]

    assert rho_i.min() > 0., 'duplicate elements in your chain'

    d_corr = float(d) / float(n) * np.sum(np.log(nu_i / rho_i))
    if not silent:
        print('  first term = %f' % d_corr)
    digamma_term = np.sum(digamma(l_i) - digamma(k_i)) / float(n)
    if not silent:
        print('  digamma term = %f' % digamma_term)
    return d_corr + digamma_term + np.log(float(m) / float(n - 1))


def fuzzy_logic_prior(x, loc, width, power):
    """
    Fuzzy logic function.
    """
    return -100 * torch.log10(1 / (1 + torch.abs((x - loc) / width)**(power)))


def log_prior(theta, bounds, powers):
    """
    Penalize parameters outside of bounds.
    """
    width = (bounds[:, 1] - bounds[:, 0]) / 2
    loc = (bounds[:, 1] + bounds[:, 0]) / 2
    index = torch.ones_like(loc) * 3
    return torch.vstack([fuzzy_logic_prior(theta[:, i], loc[i], 10 ** (index[i]
                                                                       / powers[i]) * width[i], powers[i]) for i in
                         range(len(bounds))]).mean(dim=0)


def inverse_sigmoid(x):
    """
    Inverse sigmoid function.
    """
    return torch.log(x / (1 - x))

# def transform_nmf_params(params):
#     """
#     Transform (i.e., regularize) SED parameters.
#     This might help with numerical stability.

#     We transform those params at [0, 1] using a sigmoid function.
#     We transform those parasm at [0, inf] using a log function.
#     """
#     _params = params.clone()
#     _params[:, :3] = inverse_sigmoid(_params[:, :3].clone())
#     _params[:, 3:5] = torch.log10(_params[:, 3:5].clone())
#     _params[:, 6:8] = torch.log10(_params[:, 6:8].clone())
#     _params[:, -2:-1] = torch.log10(_params[:, -2:-1].clone())
#     return _params

# def inverse_transform_nmf_params(params):
#     """
#     Inverse Transform (i.e., regularize) SED parameters.
#     This might help with numerical stability.

#     We transform those params at [0, 1] using a sigmoid function.
#     We transform those parasm at [0, inf] using a log function.
#     """
#     _params = params.clone()
#     _params[:, :3] = torch.sigmoid(_params[:, :3].clone())
#     _params[:, 3:5] = 10**(_params[:, 3:5].clone())
#     _params[:, 6:8] = 10**(_params[:, 6:8].clone())
#     _params[:, -2:-1] = 10**(_params[:, -2:-1].clone())
#     return _params


from scipy.special import erf, erfinv


def _gaussian_cdf(x, mu, sigma):
    """
    CDF of a Gaussian distribution.

    :math:`F(x) = \\frac{1}{2}(1 + erf(\\frac{x - \\mu}{\\sigma}))`
    """
    if torch.is_tensor(x):
        return 0.5 * (1 + torch.erf((x - mu) / (np.sqrt(2) * sigma)))
    else:
        return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def _inv_gaussian_cdf(x, mu, sigma):
    """
    Inverse CDF of a Gaussian distribution.

    :math:`F^{-1}(x) = \\mu + \\sigma \\sqrt{2} \\text{erfinv}(2 \\times x - 1)`

    """
    if torch.is_tensor(x):
        return mu + sigma * np.sqrt(2) * torch.erfinv(2 * x - 1)
    else:
        return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)


def cdf_transform(x, bounds):
    """
    Transform from a Gaussian (which is x) to a Uniform with bounds as input.
    """
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]


def inv_cdf_transform(x, bounds):
    """
    Transform from a Uniform with bounds (which is x) to a Gaussian
    """
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)


def transform_nmf_params(params, bounds):
    """
    Transform (i.e., regularize) SED parameters. 
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = inv_cdf_transform(_params[:, i:i + 1].clone(), bounds[i])
    return _params


def inverse_transform_nmf_params(params, bounds):
    """
    Inverse transform (i.e., regularize) SED parameters, from CDF space, to real parameter space.
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = cdf_transform(_params[:, i:i + 1].clone(), bounds[i])
    return _params


def transform_nmf_params_given_z(params, bounds, cdf_z):
    """
    Transform (i.e., regularize) SED parameters. 
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = inv_cdf_transform(_params[:, i:i + 1].clone(), bounds[i])

    _params[:, -2:-1] = cdf_z(_params[:, -2:-1].clone(), bounds[-2])
    return _params


def transform_nmf_params_given_mass(params, bounds, cdf_mass):
    """
    Transform (i.e., regularize) SED parameters. 
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = inv_cdf_transform(_params[:, i:i + 1].clone(), bounds[i])

    _params[:, -1:] = cdf_mass(_params[:, -1:].clone(), bounds[-1])
    return _params


def inverse_transform_nmf_params_given_z(params, bounds, icdf_z):
    """
    Inverse transform (i.e., regularize) SED parameters, from CDF space, to real parameter space.
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = cdf_transform(_params[:, i:i + 1].clone(), bounds[i])

    _params[:, -2:-1] = icdf_z(_params[:, -2:-1].clone(), bounds[-2])
    return _params


def inverse_transform_nmf_params_given_mass(params, bounds, icdf_mass):
    """
    Inverse transform (i.e., regularize) SED parameters, from CDF space, to real parameter space.
    This might help with the interpretation of the prior.

    Here the bounds is literally the bounds of the tophat prior in real parameter space.
    """
    _params = params.clone()
    for i in range(_params.shape[1]):
        _params[:, i:i +
                1] = cdf_transform(_params[:, i:i + 1].clone(), bounds[i])

    _params[:, -1:] = icdf_mass(_params[:, -1:].clone(), bounds[-1])
    return _params
