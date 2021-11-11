'''
Neural density estimators, build based on https://github.com/mackelab/sbi/blob/019fde2d61edbf8b4a02e034dc9c056b0d240a5c/sbi/neural_nets/flow.py#L77

But here everything is not conditioned.
'''
import torch
from torch import nn, Tensor, optim

from nflows import flows, transforms
from nflows.nn import nets
from nflows import distributions as distributions_

from sbi.utils.sbiutils import standardizing_transform
from sbi.utils.torchutils import create_alternating_binary_mask

from tqdm import trange
import pickle
import numpy as np

def build_maf(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
    ) -> nn.Module:
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
        warn(f"In one-dimensional output space, this flow is limited to Gaussians")

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

    if z_score_x: # normalize the input data
        transform_zx = standardizing_transform(batch_x)
        transform = transforms.CompositeTransform([transform_zx, transform])
        
    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net).to('cuda')

    return neural_net


def build_nsf(
    batch_x: Tensor = None,
    z_score_x: bool = True,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
    ) -> nn.Module:
    """Builds NSF to describe p(x).
    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        num_bins: Number of bins used for the splines.
        embedding_net: Optional embedding network for y.
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

        mask_in_layer = lambda i: tensor([1], dtype=uint8)
        conditioner = lambda in_features, out_features: ContextSplineMap(
            in_features, out_features, hidden_features, context_features=None
        )
        if num_transforms > 1:
            warn(
                f"You are using `num_transforms={num_transforms}`. When estimating a "
                f"1D density, you will not get any performance increase by using "
                f"multiple transforms with NSF. We recommend setting "
                f"`num_transforms=1` for faster training (see also 'Change "
                f"hyperparameters of density esitmators' here: "
                f"https://www.mackelab.org/sbi/tutorial/04_density_estimators/)."
            )

    else:
        mask_in_layer = lambda i: create_alternating_binary_mask(
            features=x_numel, even=(i % 2 == 0)
        )
        conditioner = lambda in_features, out_features: nets.ResidualNet(
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

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net).to('cuda')

    return neural_net


class NeuralDensityEstimator(object):
    """
    Neural density estimator.
    """
    def __init__(
        self,
        normalize: bool = True,
        method: str = "nsf",
        hidden_features: int = 50,
        num_transforms: int = 5,
        num_bins: int = 10,
        embedding_net: nn.Module = nn.Identity(),
        **kwargs):
        """
        Initialize neural density estimator.
        Args:
            normalize: Whether to z-score the data.
            method: Method to use for density estimation, either 'nsf' or 'maf'.
            hidden_features: Number of hidden features.
            num_transforms: Number of transforms.
            num_bins: Number of bins used for the splines.
            embedding_net: Optional embedding network for y.
            kwargs: Additional arguments that are passed by the build function but are not
                relevant for maf and are therefore ignored.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.method = method
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_bins = num_bins # only works for NSF
        self.normalize = normalize
        self.embedding_net = embedding_net
        self.train_loss_history = []

    def build(self, batch_x: Tensor, optimizer: str = "adam", **kwargs):
        """
        Build the neural density estimator based on input data.
        """
        if not torch.is_tensor(batch_x):
            batch_x = torch.tensor(batch_x, device=self.device)
        self.batch_x = batch_x

        if self.method == "maf":
            self.net = build_maf(
                batch_x=batch_x,
                z_score_x=self.normalize,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                embedding_net=self.embedding_net,
                **kwargs
            )
        elif self.method == "nsf":
            self.net = build_nsf(
                batch_x=batch_x,
                z_score_x=self.normalize,
                hidden_features=self.hidden_features,
                num_transforms=self.num_transforms,
                num_bins=self.num_bins,
                embedding_net=self.embedding_net,
                **kwargs
            )
        
        self.net.to(self.device)

        if optimizer == "adam":
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            raise ValueError(f"Unknown optimizer {optimizer}, only support 'Adam' now.")

    def train(self, n_epochs: int = 2000, display=False, suffix: str = "nde"):
        """
        Train the neural density estimator based on input data.
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
                    self.save_model(f'best_loss_model_{suffix}_{self.method}.pkl')

        if min_loss == -18:
            raise Warning('The training might be failed, try more epochs')

        if display:
            self.plot_loss()

    def sample(self, n_samples: int = 1000):
        """
        Sample according to the fitted NDE
        """
        return self.net.sample(n_samples)


    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.array(self.train_loss_history).flatten(), label='Train loss')
        plt.xlabel('Epoch')
        plt.ylabel(r'Loss = $-\sum\log(P)$')

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self = pickle.load(f)