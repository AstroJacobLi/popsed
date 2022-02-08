"""
This file contains functions to generate mock SED/spectrum, i.e., the forward model.
From 
"""
from copy import deepcopy
import numpy as np

from sedpy.observate import load_filters

from prospect.sources.constants import cosmo

#from exspect.utils import build_mock
from exspect.utils import set_sdss_lsf, load_sdss
from exspect.utils import fit_continuum, eline_mask


# --------------
# MODEL SETUP
# --------------


def build_model(uniform_priors=False, add_neb=False, add_duste=False, add_dustabs=False,
                free_neb_met=False, free_duste=False, zred_disp=0,
                snr_spec=0, continuum_optimize=False,
                **kwargs):
    """Instantiate and return a ProspectorParams model subclass. Uses WMAP9 cosmology.

    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.
    """
    from prospect.models.templates import TemplateLibrary, describe
    from prospect.models import priors, sedmodel
    has_spectrum = np.any(snr_spec > 0)

    # --- Basic parameteric SFH ---
    model_params = TemplateLibrary["parametric_sfh"]

    # --- Nebular & dust emission ---
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_neb_met:
            # Fit for independent gas metallicity
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
    if add_duste | free_duste:
        model_params.update(TemplateLibrary["dust_emission"])
        if free_duste:
            # could also adjust priors here
            model_params["duste_qpah"]["isfree"] = True
            model_params["duste_umin"]["isfree"] = True
            model_params["duste_alpha"]["isfree"] = True
            model_params["fagn"]["isfree"] = True
            model_params["agn_tau"]["isfree"] = True

    if add_dustabs:
         # Calzetti et al. (2000) attenuation curve
        model_params["dust_type"]["init"] = 2
        model_params["dust2"]["init"] = 0.0
        model_params["dust2"]["prior"] = priors.TopHat(mini=-1, maxi=0.4)
    else:
        model_params["dust_type"]["init"] = 2
        model_params["dust2"]["init"] = 0.0
        # # --- Complexify dust attenuation ---
        # # Switch to Kriek and Conroy 2013
        # model_params["dust_type"]["init"] = 4
        # # Slope of the attenuation curve, expressed as the index of the power-law
        # # that modifies the base Kriek & Conroy/Calzetti shape.
        # # I.e. a value of zero is basically calzetti with a 2175AA bump
        # model_params["dust_index"] = dict(N=1, isfree=False, init=0.0)
        # # young star dust
        # model_params["dust1"] = dict(N=1, isfree=False, init=0.0)
        # model_params["dust1_index"] = dict(N=1, isfree=False, init=-1.0)
        # model_params["dust_tesc"] = dict(N=1, isfree=False, init=7.0)

    # --- Add smoothing parameters ---
    if has_spectrum:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"]["prior"] = priors.TopHat(
            mini=150, maxi=250)
        # --- Add spectroscopic calibration ---
        if continuum_optimize:
            model_params.update(TemplateLibrary["optimize_speccal"])
            # Could change the polynomial order here
            model_params["polyorder"]["init"] = 12

    # Alter parameter values based on keyword arguments
    for p in list(model_params.keys()):
        if (p in kwargs):
            model_params[p]["init"] = kwargs[p]

    # Now set redshift free and adjust prior
    z = np.copy(model_params['zred']["init"])
    if zred_disp > 0:
        model_params['zred']["isfree"] = True
        model_params['zred']['prior'] = priors.Normal(mean=z, sigma=zred_disp)

    # Alter some priors?
    if uniform_priors:
        minit = model_params["mass"]["init"]
        model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=10)
        model_params["mass"]["prior"] = priors.TopHat(
            mini=minit/10., maxi=minit*10)

    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.98, maxi=0.19) # should be a mass-dependent ClippedNormal
    tuniv = cosmo.age(z).to("Gyr").value
    model_params["tage"]["prior"] = priors.TopHat(mini=0.1, maxi=tuniv)

    if has_spectrum & continuum_optimize:
        return sedmodel.PolySpecModel(model_params)
    else:
        return sedmodel.SpecModel(model_params)

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, add_realism=False, **extras):
    """Load the SPS object.  If add_realism is True, set up to convolve the
    library spectra to an sdss resolution
    """
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    if add_realism:
        set_sdss_lsf(sps.ssp, **extras)

    return sps

# --------------
# MOCK
# --------------


def build_mock(sps, model,
               filterset=None,
               wavelength=None,
               snr_spec=10.0, snr_phot=20., add_noise=True,
               seed=101, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated. From https://github.com/bd-j/exspect/blob/main/exspect/utils.py.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param wavelength:
        A vector

    :param snr_phot:
        The S/N of the mock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param snr_spec:
        The S/N of the mock spectroscopy.  This can also be a vector of same
        lngth as `wavelength`, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock photometry.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    """
    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which filters (and wavelengths if doing
    # spectroscopy) with which to generate mock data.
    mock = {"filters": None, "maggies": None,
            "wavelength": None, "spectrum": None}
    mock['wavelength'] = wavelength
    if filterset is not None:
        mock['filters'] = load_filters(filterset)

    # Now we get any mock params from the kwargs dict
    params = {}
    for p in model.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    model.params.update(params)
    spec, phot, mfrac = model.predict(model.theta, mock, sps=sps)

    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = np.copy(phot)
    mock['mock_params'] = deepcopy(model.params)

    # store the mock photometry
    if filterset is not None:
        pnoise_sigma = phot / snr_phot
        mock['maggies'] = phot.copy()
        mock['maggies_unc'] = pnoise_sigma
        mock['mock_snr_phot'] = snr_phot
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            pnoise = np.random.normal(0, 1, size=len(phot)) * pnoise_sigma
            mock['maggies'] += pnoise

        mock['phot_wave'] = np.array(
            [f.wave_effective for f in mock['filters']])

    # store the mock spectrum
    if wavelength is not None:
        snoise_sigma = spec / snr_spec
        mock['spectrum'] = spec.copy()
        mock['unc'] = snoise_sigma
        mock['mock_snr_spec'] = snr_spec
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            snoise = np.random.normal(0, 1, size=len(spec)) * snoise_sigma
            mock['spectrum'] += snoise

    return mock


# --------------
# OBS
# --------------


def build_obs(sps, model, dlambda_spec=2.0, wave_lo=3800, wave_hi=7000.,
              filterset=None,
              snr_spec=10., snr_phot=20., add_noise=True, seed=101,
              add_realism=False, mask_elines=False,
              continuum_optimize=False, **kwargs):
    """Build observation based on SPS and model. 

    :param wave_lo:
        The (restframe) minimum wavelength of the spectrum.

    :param wave_hi:
        The (restframe) maximum wavelength of the spectrum.

    :param dlambda_spec:
        The (restframe) wavelength sampling or spacing of the spectrum.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param snr_spec:
        S/N ratio for the spectroscopy per pixel.  scalar.

    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        Whether to add a noise realization to the spectroscopy.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.

    :param add_realism:
        If set, add a realistic S/N and instrumental dispersion based on a
        given SDSS spectrum.

    :returns obs:
        Dictionary of observational data.
    """
    # --- Make the Mock ----
    # In this demo we'll make a mock.  But we need to know which wavelengths
    # and filters to mock up.
    has_spectrum = np.any(snr_spec > 0)
    if has_spectrum:
        a = 1 + kwargs.get("zred", 0.0)
        wavelength = np.arange(wave_lo, wave_hi, dlambda_spec) * a
    else:
        wavelength = None

    if np.all(snr_phot <= 0):
        filterset = None

    # We need the models to make a mock.
    # sps = build_sps(add_realism=add_realism, **kwargs)
    # model = build_model(conintuum_optimize=continuum_optimize,
    #                     mask_elines=mask_elines, **kwargs)

    # Make spec uncertainties realistic ?
    if has_spectrum & add_realism:
        # This uses an actual SDSS spectrum to get a realistic S/N curve,
        # renormalized to have median S/N given by the snr_spec parameter
        sdss_spec, _, _ = load_sdss(**kwargs)
        snr_profile = sdss_spec['flux'] * np.sqrt(sdss_spec['ivar'])
        good = np.isfinite(snr_profile)
        snr_vec = np.interp(
            wavelength, 10**sdss_spec['loglam'][good], snr_profile[good])
        snr_spec = snr_spec * snr_vec / np.median(snr_vec)

    mock = build_mock(sps, model, filterset=filterset, snr_phot=snr_phot,
                      wavelength=wavelength, snr_spec=snr_spec,
                      add_noise=add_noise, seed=seed)

    # continuum normalize ?
    if has_spectrum & continuum_optimize:
        # This fits a low order polynomial to the spectrum and then divides by
        # that to get a continuum normalized spectrum.
        cont, _ = fit_continuum(
            mock["wavelength"], mock["spectrum"], normorder=6, nreject=3)
        cont = cont / cont.mean()
        mock["spectrum"] /= cont
        mock["unc"] /= cont
        mock["continuum"] = cont

    # Spectroscopic Masking
    if has_spectrum & mask_elines:
        mock['mask'] = np.ones(len(mock['wavelength']), dtype=bool)
        a = (1 + model.params['zred'])  # redshift the mask
        # mask everything > L(Ha)/100
        lines = np.array([3727, 3730, 3799.0, 3836.5, 3870., 3890.2, 3970,  # OII + H + NeIII
                          # H[b,g,d]  + OIII
                          4103., 4341.7, 4862.7, 4960.3, 5008.2,
                          4472.7, 5877.2, 5890.0,           # HeI + NaD
                          6302.1, 6549.9, 6564.6, 6585.3,   # OI + NII + Halpha
                          6680.0, 6718.3, 6732.7, 7137.8])  # HeI + SII + ArIII
        mock['mask'] = mock['mask'] & eline_mask(
            mock['wavelength'], lines * a, 9.0 * a)

    return mock


def _build_obs(dlambda_spec=2.0, wave_lo=3800, wave_hi=7000.,
               filterset=None,
               snr_spec=10., snr_phot=20., add_noise=True, seed=101,
               add_realism=False, mask_elines=False,
               continuum_optimize=False, **kwargs):
    """Load a mock

    :param wave_lo:
        The (restframe) minimum wavelength of the spectrum.

    :param wave_hi:
        The (restframe) maximum wavelength of the spectrum.

    :param dlambda_spec:
        The (restframe) wavelength sampling or spacing of the spectrum.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param snr_spec:
        S/N ratio for the spectroscopy per pixel.  scalar.

    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        Whether to add a noise realization to the spectroscopy.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.

    :param add_realism:
        If set, add a realistic S/N and instrumental dispersion based on a
        given SDSS spectrum.

    :returns obs:
        Dictionary of observational data.
    """
    # --- Make the Mock ----
    # In this demo we'll make a mock.  But we need to know which wavelengths
    # and filters to mock up.
    has_spectrum = np.any(snr_spec > 0)
    if has_spectrum:
        a = 1 + kwargs.get("zred", 0.0)
        wavelength = np.arange(wave_lo, wave_hi, dlambda_spec) * a
    else:
        wavelength = None

    if np.all(snr_phot <= 0):
        filterset = None

    # We need the models to make a mock.
    sps = build_sps(add_realism=add_realism, **kwargs)
    model = build_model(conintuum_optimize=continuum_optimize,
                        mask_elines=mask_elines, **kwargs)

    # Make spec uncertainties realistic ?
    if has_spectrum & add_realism:
        # This uses an actual SDSS spectrum to get a realistic S/N curve,
        # renormalized to have median S/N given by the snr_spec parameter
        sdss_spec, _, _ = load_sdss(**kwargs)
        snr_profile = sdss_spec['flux'] * np.sqrt(sdss_spec['ivar'])
        good = np.isfinite(snr_profile)
        snr_vec = np.interp(
            wavelength, 10**sdss_spec['loglam'][good], snr_profile[good])
        snr_spec = snr_spec * snr_vec / np.median(snr_vec)

    mock = build_mock(sps, model, filterset=filterset, snr_phot=snr_phot,
                      wavelength=wavelength, snr_spec=snr_spec,
                      add_noise=add_noise, seed=seed)

    # continuum normalize ?
    if has_spectrum & continuum_optimize:
        # This fits a low order polynomial to the spectrum and then divides by
        # that to get a continuum normalized spectrum.
        cont, _ = fit_continuum(
            mock["wavelength"], mock["spectrum"], normorder=6, nreject=3)
        cont = cont / cont.mean()
        mock["spectrum"] /= cont
        mock["unc"] /= cont
        mock["continuum"] = cont

    # Spectroscopic Masking
    if has_spectrum & mask_elines:
        mock['mask'] = np.ones(len(mock['wavelength']), dtype=bool)
        a = (1 + model.params['zred'])  # redshift the mask
        # mask everything > L(Ha)/100
        lines = np.array([3727, 3730, 3799.0, 3836.5, 3870., 3890.2, 3970,  # OII + H + NeIII
                          # H[b,g,d]  + OIII
                          4103., 4341.7, 4862.7, 4960.3, 5008.2,
                          4472.7, 5877.2, 5890.0,           # HeI + NaD
                          6302.1, 6549.9, 6564.6, 6585.3,   # OI + NII + Halpha
                          6680.0, 6718.3, 6732.7, 7137.8])  # HeI + SII + ArIII
        mock['mask'] = mock['mask'] & eline_mask(
            mock['wavelength'], lines * a, 9.0 * a)

    return mock

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))
