import numpy as np

from bambi.families.family import Family
from bambi.utils import get_aliased_name


# NOTE: The following methods will be discarded with the new backend
# - posterior_predictive
# - log_likelihood
# - transform_backend_kwargs
# - transform_linear_predictor
# - transform_coords
# - transform_mean


# NOTE: The following methods will be adapted with the new backend
# - get_coords
# - get_data
# - get_reference
# - get_success_level
# - transform_backend_eta
# - transform_backend_kwargs


class UnivariateFamily(Family):
    KIND = "Univariate"
    ORDINAL = False
    OUTCOME_NDIM = 1
    PARAMETER_NDIM = 1


class AsymmetricLaplace(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "b": ["log"],
        "kappa": ["log"],
        "q": ["logit", "probit", "cloglog"],
    }


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}

    def get_data(self, response):
        if response.term.data.ndim == 1:
            return response.term.data
        idx = response.levels.index(response.success)
        return response.term.data[:, idx]

    def get_success_level(self, response):
        if response.categorical:
            return get_success_level(response.term)
        return 1


class Beta(UnivariateFamily):
    """Beta family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    SUPPORTED_LINKS = {"mu": ["logit", "probit", "cloglog"], "kappa": ["log"]}


class BetaBinomial(UnivariateFamily):
    """BetaBinomial family

    It uses the mean (mu) and sample size (kappa) parametrization of the Beta distribution.
    """

    SUPPORTED_LINKS = {"mu": ["logit", "probit", "cloglog"], "kappa": ["log"]}


class Binomial(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}


class Categorical(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}
    INVLINK_KWARGS = {"axis": -1}
    PARAMETER_NDIM = 2

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]

    def get_coords(self, response):
        name = get_aliased_name(response) + "_reduced_dim"
        return {name: [level for level in response.levels if level != response.reference]}

    def get_reference(self, response):
        return get_reference_level(response.term)


class Cumulative(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["logit", "probit", "cloglog"], "threshold": ["identity"]}
    PARAMETER_NDIM = 2
    ORDINAL = True

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]


class Exponential(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"]}


class Gamma(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "alpha": ["log"]}


class Gaussian(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"]}


class HurdleGamma(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class HurdleLogNormal(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "sigma": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class HurdleNegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "cloglog"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class HurdlePoisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"], "psi": ["logit", "probit", "cloglog"]}


class NegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "cloglog"], "alpha": ["log"]}


class Laplace(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "b": ["log"]}


class Poisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"]}


class StoppingRatio(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["logit", "probit", "cloglog"], "threshold": ["identity"]}
    PARAMETER_NDIM = 2
    ORDINAL = True

    def get_data(self, response):
        return np.nonzero(response.term.data)[1]


class StudentT(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log", "inverse"], "sigma": ["log"], "nu": ["log"]}


class VonMises(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity"], "kappa": ["log"]}


class Wald(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["inverse", "inverse_squared", "identity", "log"], "lam": ["log"]}


class Weibull(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["log", "identity", "inverse"], "alpha": ["log"]}


class ZeroInflatedBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {
        "p": ["identity", "logit", "probit", "cloglog"],
        "psi": ["logit", "probit", "cloglog"],
    }


class ZeroInflatedNegativeBinomial(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "cloglog"],
        "alpha": ["log"],
        "psi": ["logit", "probit", "cloglog"],
    }


class ZeroInflatedPoisson(UnivariateFamily):
    SUPPORTED_LINKS = {"mu": ["identity", "log"], "psi": ["logit", "probit", "cloglog"]}


# pylint: disable = protected-access
def get_success_level(term):
    """Returns the success level of a categorical term

    Whenever the concept of "success level" does not apply, it returns `None`.
    """
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return term.components[0].reference

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]


# pylint: disable = protected-access
def get_reference_level(term):
    """Returns the reference level of a categorical term

    Whenever the concept of "reference level" does not apply, it returns `None`.
    """
    if term.kind != "categoric":
        return None

    if term.levels is None:
        return None

    levels = term.levels
    intermediate_data = term.components[0]._intermediate_data
    if hasattr(intermediate_data, "_contrast"):
        return intermediate_data._contrast.reference

    return levels[0]
