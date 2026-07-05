import logging
import traceback
import warnings
from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pymc as pm
from pymc.util import get_default_varnames

from bambi.backend.pymc.coords import coords_from_response
from bambi.backend.pymc.parameters import build_conditional_parameter, build_marginal_parameter
from bambi.backend.pymc.terms import build_response_term

_log = logging.getLogger("bambi")


__version__ = version("bambi")


_SUPPORTED_METHODS = {"pymc", "numpyro", "blackjax", "nutpie", "vi", "laplace"}
_DEPRECATION_MAP = {
    "mcmc": "pymc",
    "nuts_numpyro": "numpyro",
    "numpyro_nuts": "numpyro",
    "nuts_blackjax": "blackjax",
    "blackjax_nuts": "blackjax",
}


class PyMCModel:
    def __init__(self, model):
        """_summary_

        Parameters
        ----------
        model : bmb.Model
            The bambi model specification.
        """
        self.model = None
        self.spec = model

    def build(self):
        ## Global process:
        # 1. Build dims and coordinates
        # 2. Instantiate model
        # 3. Create data containers

        ## ConditionalParameter process
        # 1. Build RVs
        # 2. Manipulate data containers and RVs (grab stuff from PyMC model)
        # 3. Create deterministics (grab stuff from PyMC model)
        # 1. Build dims, coordinates, and term
        # 3. Build random variables
        # 4. Build response, again grabbing stuff from the PyMC model.
        response_coords_data, response_coords, response_coords_reduced = coords_from_response(
            self.spec.response_term, self.spec.family
        )

        model = pm.Model(coords=response_coords_data | response_coords | response_coords_reduced)
        model.__bambi_attrs__ = {
            "response_ndim": self.spec.family.RESPONSE_NDIM,
            "response_coords_data": response_coords_data,
            "response_coords": response_coords,
            "response_coords_reduced": response_coords_reduced,
        }

        marginal_parameters = {}
        conditional_parameters = {}
        for name, parameter in self.spec.marginal_parameters.items():
            marginal_parameters[name] = build_marginal_parameter(parameter, self.spec.family, model)

        for name, parameter in self.spec.conditional_parameters.items():
            conditional_parameters[name] = build_conditional_parameter(
                parameter, self.spec.family, model
            )

        build_response_term(
            term=self.spec.response_term,
            parameters=marginal_parameters | conditional_parameters,
            family=self.spec.family,
            model=model,
        )

        self.model = model

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_response_params=False,
        inference_method="pymc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        nuts=None,
        **kwargs,
    ):
        """Run PyMC sampler."""
        inference_method = inference_method.lower()

        # Handle deprecated inference methods
        if inference_method in _DEPRECATION_MAP:
            new_method = _DEPRECATION_MAP[inference_method]
            warnings.warn(
                f"'{inference_method}' has been replaced by '{new_method}' and will be "
                "removed in a future release.",
                category=FutureWarning,
            )
            inference_method = new_method

        # Validate the inference method
        if inference_method not in _SUPPORTED_METHODS:
            # Use sorted() for a predictable, user-friendly error message
            supported = ", ".join(sorted(_SUPPORTED_METHODS))
            raise ValueError(
                f"'{inference_method}' is not a supported inference method. "
                f"Must be one of: {supported}"
            )

        # Ensure the appropriate dependencies are installed for the selected inference method
        self._check_dependencies(inference_method)

        # NOTE: Methods return different types of objects (idata, approximation, and dictionary)
        if inference_method == "vi":
            result = self._run_vi(random_seed=random_seed, **kwargs)
        elif inference_method == "laplace":
            result = self._run_laplace(
                draws=draws,
                omit_offsets=omit_offsets,
                include_response_params=include_response_params,
            )
        else:
            result = self._run_mcmc(
                draws=draws,
                tune=tune,
                discard_tuned_samples=discard_tuned_samples,
                omit_offsets=omit_offsets,
                include_response_params=include_response_params,
                init=init,
                n_init=n_init,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                nuts=nuts,
                sampler_backend=inference_method,
                **kwargs,
            )

        self.fit = True
        return result

    def _run_mcmc(
        self,
        draws,
        tune,
        discard_tuned_samples,
        omit_offsets,
        include_response_params,
        init,
        n_init,
        chains,
        cores,
        random_seed,
        nuts,
        sampler_backend,
        **kwargs,
    ):
        # Don't include the parameters of the likelihood, which are deterministics.
        # They can take lot of space in the trace and increase RAM requirements.
        vars_to_sample = get_default_varnames(
            self.model.unobserved_value_vars, include_transformed=False
        )
        vars_to_sample = [variable.name for variable in vars_to_sample]

        for name, variable in self.model.named_vars.items():
            is_likelihood_param = name in self.spec.family.likelihood.params
            is_deterministic = variable in self.model.deterministics
            if is_likelihood_param and is_deterministic:
                vars_to_sample.remove(name)

        # pm.sample routes nuts settings via kwargs.pop("nuts", {}); only inject when provided
        # to avoid passing nuts=None which causes pm.sample's internal nuts_kwargs.copy() to fail.
        if nuts is not None:
            kwargs["nuts"] = nuts

        with self.model:
            try:
                idata = pm.sample(
                    draws=draws,
                    tune=tune,
                    discard_tuned_samples=discard_tuned_samples,
                    init=init,
                    n_init=n_init,
                    chains=chains,
                    cores=cores,
                    random_seed=random_seed,
                    var_names=vars_to_sample,
                    nuts_sampler=sampler_backend,
                    **kwargs,
                )
            except (RuntimeError, ValueError):
                if "ValueError: Mass matrix contains" in traceback.format_exc() and init == "auto":
                    _log.info(
                        "\nThe default initialization using init='auto' has failed, trying to "
                        "recover by switching to init='adapt_diag'",
                    )
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        discard_tuned_samples=discard_tuned_samples,
                        init="adapt_diag",
                        n_init=n_init,
                        chains=chains,
                        cores=cores,
                        random_seed=random_seed,
                        var_names=vars_to_sample,
                        nuts_sampler=sampler_backend,
                        **kwargs,
                    )
                else:
                    raise

        # Before doing anything, make sure we compute deterministics.
        # But, don't include those determinisics for parameters of the likelihood.
        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = __version__

        if omit_offsets:
            offset_vars = [var for var in idata.posterior.data_vars if var.endswith("_offset")]
            idata.posterior = idata.posterior.drop_vars(offset_vars)

        if include_response_params:
            self.spec.predict(idata)

        return idata

    def _run_vi(self, random_seed, **kwargs):
        with self.model:
            self.vi_approx = pm.fit(random_seed=random_seed, **kwargs)
        return self.vi_approx

    def _run_laplace(self, draws, omit_offsets, include_response_params):
        """Fit a model using a Laplace approximation.

        Mainly for pedagogical use, provides reasonable results for approximately Gaussian
        posteriors. The approximation can be very poor for some models  like hierarchical ones.

        Parameters
        ----------
        draws : int
            The number of samples to draw from the posterior distribution.
        omit_offsets : bool
            Omits offset terms in the `InferenceData` object returned when the model includes
            group specific effects.
        include_response_params : bool
            Compute the posterior of the mean response.

        Returns
        -------
        An ArviZ's InferenceData object.
        """
        with self.model:
            maps = pm.find_MAP()
            n_maps = deepcopy(maps)

            # Remove deterministics for parent parameters
            n_maps = {
                key: value
                for key, value in n_maps.items()
                if key not in self.spec.family.likelihood.params
            }

            for m in maps:
                if pm.util.is_transformed_name(m):
                    untransformed_name = pm.util.get_untransformed_name(m)
                    if untransformed_name in n_maps:
                        n_maps.pop(untransformed_name)

            hessian = pm.find_hessian(n_maps)

        if np.linalg.det(hessian) == 0:
            raise np.linalg.LinAlgError("Singular matrix. Use mcmc or vi method")

        cov = np.linalg.inv(hessian)
        modes = np.concatenate([np.atleast_1d(v) for v in n_maps.values()])

        samples = np.random.multivariate_normal(modes, cov, size=draws)

        idata = _posterior_samples_to_idata(samples, self.model)
        idata = self._clean_results(idata, omit_offsets, include_response_params)
        return idata

    def _check_dependencies(self, inference_method):
        """Dependency checking given the selected inference method."""
        required_packages = {
            "numpyro": ["numpyro", "jax"],
            "blackjax": ["blackjax", "jax"],
            "nutpie": ["nutpie"],
        }

        if inference_method in required_packages:
            missing = []
            for package in required_packages[inference_method]:
                try:
                    __import__(package)
                except ImportError:
                    missing.append(package)

            if missing:
                raise ImportError(
                    f"'{inference_method}' requires package(s): {', '.join(missing)}. "
                )


def _posterior_samples_to_idata(samples, model):
    """Create InferenceData from samples

    Parameters
    ----------
    samples : array
        Posterior samples
    model : PyMC model

    Returns
    -------
    An ArviZ's InferenceData object.
    """
    initial_point = model.initial_point()
    variables = model.value_vars

    var_info = {}
    for name, value in initial_point.items():
        var_info[name] = (value.shape, value.size)

    length_pos = len(samples)
    varnames = [v.name for v in variables]

    with model:
        strace = pm.backends.ndarray.NDArray(name=model.name)  # pylint:disable=no-member
        strace.setup(length_pos, 0)
    for i in range(length_pos):
        value = []
        size = 0
        for varname in varnames:
            shape, new_size = var_info[varname]
            var_samples = samples[i][size : size + new_size]
            value.append(var_samples.reshape(shape))
            size += new_size
        strace.record(point=dict(zip(varnames, value)))

    idata = pm.to_inference_data(pm.backends.base.MultiTrace([strace]), model=model)
    return idata
