import inspect
import functools

import pymc as pm
import pytensor.tensor as pt

from bambi.backend.new_pymc.links import (
    cloglog,
    identity,
    inverse_squared,
    logit,
    probit,
)
from pytensor.tensor.special import softmax

MAPPING = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical}

INVERSE_LINKS = {
    "cloglog": cloglog,
    "identity": identity,
    "inverse_squared": inverse_squared,
    "inverse": pt.reciprocal,
    "log": pt.exp,
    "logit": logit,
    "probit": probit,
    "softmax": functools.partial(softmax, axis=-1),
}


def get_distribution(dist):
    """Return a PyMC distribution."""
    if isinstance(dist, str):
        if dist in MAPPING:
            dist = MAPPING[dist]
        elif hasattr(pm, dist):
            dist = getattr(pm, dist)
        else:
            raise ValueError(f"The Distribution '{dist}' was not found in PyMC")
    return dist


def get_distribution_from_prior(prior):
    if prior.dist is not None:
        distribution = prior.dist
    else:
        distribution = get_distribution(prior.name)
    return distribution


def get_distribution_from_likelihood(likelihood):
    """
    It works because both `Prior` and `Likelihood` instances have a `name` and a `dist` argument.
    """
    return get_distribution_from_prior(likelihood)


def make_weighted_logp(dist: pm.Distribution):
    """Create a function to compute a weighted logp

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution for which we want to get the weighted logp.

    Returns
    -------
    A function that computes the weighted logp.
    """

    def logp(value, *dist_params, weights):
        weights = pt.as_tensor_variable(weights)
        return weights * pm.logp(dist.dist(*dist_params), value)

    return logp


def get_dist_args(dist: pm.Distribution) -> list[str]:
    """Get the argument names of a PyMC distribution

    The argument names are the names of the parameters of the distribution.

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution for which we want to extract the argument names.

    Returns
    -------
    list[str]
        The names of the arguments.
    """
    # Get all args but the first one which is usually 'cls'
    return inspect.getfullargspec(dist.dist).args[1:]


def create_cdist(dist: pm.Distribution):
    def fun(*params):
        *dist_params, size = params
        return dist.dist(*dist_params, size=size)

    return fun


# pylint: disable=bare-except
# pylint: disable=protected-access
def make_weighted_distribution(dist: pm.Distribution):
    wlogp = make_weighted_logp(dist)
    dist_args = get_dist_args(dist)

    try:
        dname = dist.rv_op._print_name[0]
    except:
        dname = "Dist"

    cdist = create_cdist(dist)
    class_name = f"Weighted{dname}"

    class WeightedDistribution:
        # We pass 'logp' to get the weighted logp, and we pass 'dist' to make sure
        # the random draws are generated using the correct parameter values.
        # Distribution.dist is the method that handles the parameters and with this approach
        # we are sure that we use it.
        def __new__(cls, name, weights, **kwargs):
            # Get parameter values in the order required by the distribution as they are passed
            # by position to `pm.CustomDist`
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            return pm.CustomDist(
                name,
                *dist_params,
                logp=functools.partial(wlogp, weights=weights),
                dist=cdist,
                class_name=class_name,
                **kwargs,
            )

        @classmethod
        def dist(cls, **kwargs):
            dist_params = [kwargs.pop(arg) for arg in dist_args if arg in kwargs]
            weights = 1 if "weights" not in kwargs else kwargs.pop("weights")
            return pm.CustomDist.dist(
                *dist_params,
                logp=functools.partial(wlogp, weights=weights),
                dist=cdist,
                class_name=class_name,
                **kwargs,
            )

    return WeightedDistribution
