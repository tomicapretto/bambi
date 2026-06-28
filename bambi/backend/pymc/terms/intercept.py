import numpy as np
import pymc as pm

from bambi.backend.pymc.utils import get_distribution_from_prior


def build_intercept_term(term, data_mean, common_params, model):
    coords = model.__bambi_attrs__["response_coords_reduced"]
    dims = tuple(coords)
    param_shape = tuple(len(coord) for coord in coords.values())
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        if data_mean is not None:
            # TODO: Automatic transformation of intercept prior such that users can pass it cleanly.
            # Covariates are centered, thus we uncenter the intercept.
            rv = dist(term.label + "_centered", **kwargs, dims=dims)
            pm.Deterministic(term.label, rv - pm.math.sum(data_mean * common_params), dims=dims)
        else:
            rv = dist(term.label, **kwargs, dims=dims)

    return rv
