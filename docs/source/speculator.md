# Speculator: Galaxy Spectrum Emulator

`popsed.speculator` is the main module of the Speculator.

Add descriptions of the emulator here. Including how we train it. 
Maybe also nice to add the following.

One thing confuses me a lot is the unit of the spectrum. To convert from ``L_sun/Hz`` to ``L_sun/AA``, multiply ``L_sun/Hz`` by ``lightspeed / wave**2``.
To convert ``L_sun/Hz`` to ``erg/s/cm^2/AA`` at 10 pc, multiply by ``to_cgs_at_10pc``.


```{eval-rst}
.. toctree::
    :maxdepth: 2

    popsed.speculator

Speculator
------------

.. currentmodule:: popsed.speculator.Speculator

.. autoclass:: popsed.speculator.Speculator

.. autosummary::
  :toctree: _autosummary

    __init__
    _build_params_prior
    _build_distance_interpolator
    _calc_transmission
    _parse_noise_model
    load_data
    train
    plot_loss
    transform
    save_model
    _predict_pca
    _predict_spec_with_mass_restframe
    _predict_spec_with_mass_redshift
    predict_spec
    _predict_mag_with_mass_redshift


SuperSpeculator
----------------
The ``SuperSpeculator`` class is used to combine several individual ``Speculator`` instances into a single model. Single ``Speculator`` is trained for specific wavelength range.

.. currentmodule:: popsed.speculator.SuperSpeculator

.. autoclass:: popsed.speculator.SuperSpeculator
    :show-inheritance:

.. autosummary::
  :toctree: _autosummary

    __init__
    _predict_spec_restframe
    _predict_spec_with_mass_redshift
    _predict_mag_with_mass_redshift
    _predict_mag_with_mass_redshift_batch


Other classes
----------------

.. currentmodule:: popsed.speculator.StandardScaler
.. autoclass:: popsed.speculator.StandardScaler
.. autosummary:: 
    :toctree: _autosummary

    __init__
    fit
    transform
    inverse_transform

.. currentmodule:: popsed.speculator.SpectrumPCA
.. autoclass:: popsed.speculator.SpectrumPCA
.. autosummary:: 
   :toctree: _autosummary

    __init__
    scale_spectra
    train_pca
    inverse_transform
    validate_pca_basis
    save

.. autoclass:: CustomActivation

.. autoclass:: FC

.. autoclass:: Network

```