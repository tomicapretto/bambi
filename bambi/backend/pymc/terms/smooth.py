import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.utils import get_distribution_from_prior


def build_smooth_coefficients(term, dims):
    """Build one Normal coefficient vector with a shared scale for its curved part."""
    name = term.label
    null_count = term.null_space_dimension
    curved_count = term.shape[1] - null_count
    null_prior, scale_prior = term.prior["null"], term.prior["sigma"]
    tau = get_distribution_from_prior(scale_prior)(f"{name}_sigma", **scale_prior.args)
    mu = pt.concatenate(
        [pt.broadcast_to(null_prior.args["mu"], (null_count,)), pt.zeros(curved_count)]
    )
    sigma = pt.concatenate(
        [pt.broadcast_to(null_prior.args["sigma"], (null_count,)), pt.repeat(tau, curved_count)]
    )
    return pm.Normal(name, mu=mu, sigma=sigma, dims=dims)
