import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.types import CoefSpec, Constraint


def build_intercept_term(
    term,
    data_mean: np.ndarray | None,
    common_params: pt.Variable | None,
    coef_spec: CoefSpec,
    model: pm.Model,
) -> pt.Variable:
    # NOTE (idea): Transform the intercept prior such that users can pass it cleanly.
    coords = {}
    if coef_spec.ndim > 0:
        if coef_spec.constraint == Constraint.REFERENCE:
            coords = model.__bambi_attrs__["response_coords_reduced"]
        else:
            coords = model.__bambi_attrs__["response_coords"]

    dims = tuple(coords)
    param_shape = tuple(len(coord) for coord in coords.values())
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        if data_mean is not None and common_params is not None:
            # Covariates are centered, thus we uncenter the intercept.
            rv = dist(term.label + "_centered", **kwargs, dims=dims)
            data_mean = data_mean.reshape((-1,))
            if common_params.ndim == 1:
                offset = pm.math.sum(data_mean * common_params)
            else:
                offset = pt.dot(data_mean, common_params)
            pm.Deterministic(term.label, rv - offset, dims=dims)
        else:
            rv = dist(term.label, **kwargs, dims=dims)

    return rv
