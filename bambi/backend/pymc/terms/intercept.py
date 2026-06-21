import numpy as np

from bambi.backend.pymc.utils import get_distribution_from_prior


def build_intercept_term(term, model):
    coords = model.__bambi_attrs__["response_coords_reduced"]
    dims = tuple(coords)
    param_shape = tuple(len(coord) for coord in coords.values())
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        rv = dist(term.label, **kwargs, dims=dims)
    return rv
