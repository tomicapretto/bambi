from bambi.families.family import Family
from bambi.transformations import transformations_namespace
from bambi.utils import extract_argument_names


class UnivariateFamily(Family):
    KIND = "Univariate"
    ORDINAL = False
    PARAMETER_NDIM = 1
    RESPONSE_NDIM = 1
    DATA_TYPE = "numeric"


class AsymmetricLaplace(UnivariateFamily):
    SUPPORTED_LINKS = {
        "mu": ["identity", "log", "inverse"],
        "b": ["log"],
        "kappa": ["log"],
        "q": ["logit", "probit", "cloglog"],
    }


class Bernoulli(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["identity", "logit", "probit", "cloglog"]}
    DATA_TYPE = "binary"


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
    PARAMETER_NDIM = 2
    DATA_TYPE = "categorical"


class Cumulative(UnivariateFamily):
    SUPPORTED_LINKS = {"p": ["logit", "probit", "cloglog"], "threshold": ["identity"]}
    PARAMETER_NDIM = 2
    ORDINAL = True
    DATA_TYPE = "categorical"


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
    DATA_TYPE = "categorical"


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


class MultivariateFamily(Family):
    KIND = "Multivariate"
    ORDINAL = False
    DATA_TYPE = "numeric"
    RESPONSE_NDIM = 2


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"p": ["softmax"]}

    def get_levels(self, response):
        labels = extract_argument_names(response.name, list(transformations_namespace))
        if labels:
            return labels
        return [str(level) for level in range(response.data.shape[1])]


class DirichletMultinomial(MultivariateFamily):
    SUPPORTED_LINKS = {"a": ["log"]}

    def get_levels(self, response):
        levels = extract_argument_names(response.name, list(transformations_namespace))
        if levels:
            return levels
        return [str(level) for level in range(response.data.shape[1])]
