# Adapted from PROVABGS: https://github.com/changhoonhahn/provabgs
import os
import h5py
import numpy as np
import scipy.stats as stat
import scipy.optimize as op


class Prior(object):
    ''' base class for prior
    '''

    def __init__(self, label=None):
        self.ndim = None
        self.ndim_sampling = None
        self.label = label
        self._random = np.random.mtrand.RandomState()

    def transform(self, tt):
        ''' Some priors require transforming the parameter space for sampling
        (e.g. Dirichlet priors). For most priors, this transformation will return 
        the same value  
        '''
        return tt

    def untransform(self, tt):
        ''' Some priors require transforming the parameter space for sampling
        (e.g. Dirichlet priors). For most priors, this transformation will return 
        the same value  
        '''
        return tt

    def lnPrior(self, theta):
        ''' evaluate the log prior at theta
        '''
        pass


class FlatDirichletPrior(Prior):
    '''
    Flat Dirichlet prior
    '''

    def __init__(self, ndim, label=None):
        super().__init__(label=label)
        self.ndim = ndim
        self.ndim_sampling = ndim - 1

    def transform(self, tt):
        ''' warped manifold transformation as specified in Betancourt (2013).
        This function transforms samples from a uniform distribution to a
        Dirichlet distribution .

        x_i = (\prod\limits_{k=1}^{i-1} z_k) * f 

        f = 1 - z_i         for i < m
        f = 1               for i = m 

        Parameters
        ----------
        tt : array_like[N,m-1]
            N samples drawn from a (m-1)-dimensional uniform distribution 

        Returns
        -------
        tt_d : array_like[N,m]
            N transformed samples drawn from a m-dimensional dirichlet
            distribution 

        Reference
        ---------
        * Betancourt(2013) - https://arxiv.org/pdf/1010.3436.pdf
        '''
        assert tt.shape[-1] == self.ndim_sampling
        tt_d = np.empty(tt.shape[:-1] + (self.ndim,))

        tt_d[..., 0] = 1. - tt[..., 0]
        for i in range(1, self.ndim_sampling):
            tt_d[..., i] = np.prod(tt[..., :i], axis=-1) * (1. - tt[..., i])
        tt_d[..., -1] = np.prod(tt, axis=-1)
        return tt_d

    def untransform(self, tt_d):
        ''' reverse the warped manifold transformation 
        '''
        assert tt_d.shape[-1] == self.ndim
        tt = np.empty(tt_d.shape[:-1] + (self.ndim_sampling,))

        tt[..., 0] = 1. - tt_d[..., 0]
        for i in range(1, self.ndim_sampling):
            tt[..., i] = 1. - (tt_d[..., i] / np.prod(tt[..., :i], axis=-1))
        return tt

    def lnPrior(self, theta):
        ''' evaluate the prior at theta. We assume theta here is
        *untransformed* --- i.e. sampled from a uniform distribution. 

        Parameters
        ----------
        theta : array_like[m,]
            m-dimensional set of parameters 

        Return
        ------
        prior : float
            prior evaluated at theta
        '''
        if np.all(theta <= np.ones(self.ndim_sampling)) and np.all(theta >= np.zeros(self.ndim_sampling)):
            return 0.
        else:
            return -np.inf

    def append(self, *arg, **kwargs):
        raise ValueError("appending not supproted")

    def sample(self):
        return np.array([self._random.uniform(0., 1.) for i in range(self.ndim_sampling)])


class UniformPrior(Prior):
    ''' 
    Uniform tophat prior
    '''

    def __init__(self, _min, _max, label=None):
        super().__init__(label=label)

        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = self.min.shape[0]
        self.ndim_sampling = self.ndim
        assert self.min.shape[0] == self.max.shape[0]
        assert np.all(self.min <= self.max)

    def lnPrior(self, theta):
        if np.all(theta <= self.max) and np.all(theta >= self.min):
            return 0.
        else:
            return -np.inf

    def sample(self):
        return np.array([self._random.uniform(mi, ma) for (mi, ma) in zip(self.min, self.max)])


class LogUniformPrior(Prior):
    '''
    Log uniform tophat prior
    '''

    def __init__(self, _min, _max, label=None):
        super().__init__(label=label)

        self.min = np.atleast_1d(_min)
        self.max = np.atleast_1d(_max)
        self.ndim = self.min.shape[0]
        self.ndim_sampling = self.ndim
        assert self.min.shape[0] == self.max.shape[0]
        assert np.all(self.min <= self.max)

    def lnPrior(self, theta):
        if np.all(theta <= self.max) and np.all(theta >= self.min):
            return 0.
        else:
            return -np.inf

    def sample(self):
        return np.array([10**self._random.uniform(np.log10(mi), np.log10(ma)) for mi, ma in zip(self.min, self.max)])


class GaussianPrior(Prior):
    '''
    Gaussian prior 
    '''

    def __init__(self, mean, covariance, label=None):
        super().__init__(label=label)

        self.mean = np.atleast_1d(mean)
        self.covariance = np.atleast_1d(covariance)
        self.ndim = self.mean.shape[0]
        self.ndim_sampling = self.ndim
        assert self.mean.shape[0] == self.covariance.shape[0]
        self.multinorm = stat.multivariate_normal(self.mean, self.covariance)

    def lnPrior(self, theta):
        return self.multinorm.logpdf(theta)

    def sample(self):
        return np.atleast_1d(self.multinorm.rvs())


class BiGaussianPrior(Prior):
    '''
    Two Gaussians
    '''

    def __init__(self, mean, covariance, p1=0.3, label=None):
        super().__init__(label=label)

        self.mean = np.atleast_1d(mean)
        self.covariance = np.atleast_1d(covariance)
        self.ndim = self.mean.shape[0]
        self.ndim_sampling = self.ndim
        assert self.mean.shape[0] == self.covariance.shape[0]
        self.multinorm = stat.multivariate_normal(self.mean, self.covariance)
        self.p1 = p1

    def lnPrior(self, theta):
        return self.multinorm.logpdf(theta)

    def sample(self):
        if self._random.uniform() < self.p1:
            return np.atleast_1d(self.multinorm.rvs()[0])
        else:
            return np.atleast_1d(self.multinorm.rvs()[1])


class TruncatedNormalPrior(Prior):
    '''
    Truncated normal prior 
    '''

    def __init__(self, a, b, mean, sigma, label=None):
        '''
        Parameters
        ----------
        a : float, lower truncation limit
        b : float, upper truncation limit
        mean: float, mean of the normal distribution
        covariance: float, covariance of the normal distribution
        '''
        super().__init__(label=label)
        self.mean = np.atleast_1d(mean)
        self.sigma = np.atleast_1d(sigma)
        self.a = (a - self.mean) / self.sigma
        self.b = (b - self.mean) / self.sigma
        self.ndim = self.mean.shape[0]
        self.ndim_sampling = self.ndim
        assert self.mean.shape[0] == self.sigma.shape[0]
        self.truncnorm = stat.truncnorm(
            a=self.a, b=self.b, loc=self.mean, scale=self.sigma)

    def lnPrior(self, theta):
        return self.truncnorm.logpdf(theta)

    def sample(self):
        return np.atleast_1d(self.truncnorm.rvs())


def load_priors(list_of_prior_obj):
    '''
    Load a list of `Prior` class objects into a PriorSeq object

    Example
    -------
    load_priors([
        Infer.FlatDirichletPrior(ncomp, label='sed'),       # flat dirichilet priors
        Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 4.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1 
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-3., 1., label='sed'),     # uniform priors on dust_index 
        Infer.UniformPrior(0., 0.6, label='sed')       # uniformly sample redshift
        ])
    '''
    return PriorSeq(list_of_prior_obj)


class PriorSeq(object):
    '''
    Immutable sequence of priors that are assumed to be statistically independent. 

    Parmaeter
    ---------
    list_of_priors : array_like [Npriors,]
        list of `infer.Prior` objects
    '''

    def __init__(self, list_of_priors):
        self.list_of_priors = list_of_priors

    def lnPrior(self, theta):
        ''' evaluate the log prior at theta 
        '''
        theta = np.atleast_2d(theta)

        i = 0
        lnp_theta = 0.
        for prior in self.list_of_priors:
            _lnp_theta = prior.lnPrior(theta[:, i:i + prior.ndim_sampling])
            if not np.isfinite(_lnp_theta):
                return -np.inf

            lnp_theta += _lnp_theta
            i += prior.ndim_sampling

        return lnp_theta

    def sample(self):
        ''' sample the prior 
        '''
        samp = []
        for prior in self.list_of_priors:
            samp.append(prior.sample())
        return np.concatenate(samp)

    def transform(self, tt):
        ''' transform theta 
        '''
        tt_p = np.empty(tt.shape[:-1] + (self.ndim,))

        i, _i = 0, 0
        for prior in self.list_of_priors:
            tt_p[..., i:i +
                 prior.ndim] = prior.transform(tt[..., _i:_i + prior.ndim_sampling])
            i += prior.ndim
            _i += prior.ndim_sampling
        return tt_p

    def untransform(self, tt):
        ''' transform theta 
        '''
        tt_p = np.empty(tt.shape[:-1] + (self.ndim_sampling,))

        i, _i = 0, 0
        for prior in self.list_of_priors:
            tt_p[..., i:i +
                 prior.ndim_sampling] = prior.untransform(tt[..., _i:_i + prior.ndim])
            i += prior.ndim_sampling
            _i += prior.ndim
        return tt_p

    def separate_theta(self, theta, labels=None):
        ''' separate theta based on label
        '''
        lbls = np.concatenate([np.repeat(prior.label, prior.ndim)
                               for prior in self.list_of_priors])

        output = []
        for lbl in labels:
            islbl = (lbls == lbl)
            output.append(theta[islbl])
        return output

    def append(self, another_list_of_priors):
        ''' append more Prior objects to the sequence 
        '''
        # join list
        self.list_of_priors += another_list_of_priors
        return None

    @property
    def ndim(self):
        # update ndim
        return np.sum([prior.ndim for prior in self.list_of_priors])

    @property
    def ndim_sampling(self):
        # update ndim
        return np.sum([prior.ndim_sampling for prior in self.list_of_priors])

    @property
    def labels(self):
        return np.array([prior.label for prior in self.list_of_priors])

    @property
    def range(self):
        ''' range of the priors 
        '''
        prior_min, prior_max = [], []
        for prior in self.list_of_priors:
            if isinstance(prior, UniformPrior):
                _min = prior.min
                _max = prior.max
            elif isinstance(prior, LogUniformPrior):
                _min = prior.min
                _max = prior.max
            elif isinstance(prior, FlatDirichletPrior):
                _min = np.zeros(prior.ndim)
                _max = np.ones(prior.ndim)
            elif isinstance(prior, GaussianPrior):
                _min = prior.mean - 3. * np.sqrt(np.diag(prior.covariance))
                _max = prior.mean + 3. * np.sqrt(np.diag(prior.covariance))
            else:
                raise ValueError
            prior_min.append(np.atleast_1d(_min))
            prior_max.append(np.atleast_1d(_max))
        prior_min = np.concatenate(prior_min)
        prior_max = np.concatenate(prior_max)
        return prior_min, prior_max
