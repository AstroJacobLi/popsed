# Neural Density Estimator

`popsed.nde` is the main module of the neural density estimator.



```{eval-rst}


NeuralDensityEstimator
-----------------------

.. currentmodule:: popsed.nde.NeuralDensityEstimator

.. autoclass:: popsed.nde.NeuralDensityEstimator

.. autosummary::
  :toctree: _autosummary

    __init__
    build
    _train
    sample
    plot_loss
    save_model


WassersteinNeuralDensityEstimator
---------------------------------

.. currentmodule:: popsed.nde.WassersteinNeuralDensityEstimator

.. autoclass:: popsed.nde.WassersteinNeuralDensityEstimator
    :show-inheritance:

.. autosummary::
  :toctree: _autosummary

    __init__
    build
    load_validation_data
    _get_loss_NMF
    train
    goodness_of_fit


Other classes
----------------

.. currentmodule:: popsed.nde

.. autosummary::
    
    :toctree: _autosummary
    cdf_transform
    inv_cdf_transform
    transform_nmf_params
    inverse_transform_nmf_params

```