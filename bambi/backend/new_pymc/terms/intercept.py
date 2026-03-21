import numpy as np

from bambi.backend.new_pymc.utils import get_distribution_from_prior


def build_intercept_term(term, model):
    coords = {k: v for k, v in model.__bambi_attrs__["output_coords"].items() if k != "__obs__"}
    dims = tuple(coords)
    param_shape = tuple(len(coord) for coord in coords.values()) or (1,)
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)
    return dist(term.name, **kwargs, dims=dims, model=model)
