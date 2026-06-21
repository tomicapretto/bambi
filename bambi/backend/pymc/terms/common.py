import numpy as np
import pymc as pm

from bambi.backend.pymc.coords import coords_from_common
from bambi.backend.pymc.utils import get_distribution_from_prior


def build_common_term(term, model):
    """_summary_

    Parameters
    ----------
    term : bambi.terms.CommonTerm
        ...
    model : pymc.Model
        ...

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    data_name = f"{term.label}_data"
    param_name = term.label
    coords = coords_from_common(term)

    # Register coords
    if data_name not in model or param_name not in model:
        model.add_coords(coords)

    # Register data
    if data_name not in model:
        # Reshapes to 2D if not already
        data_dims = ("__obs__", *coords)
        pm.Data(data_name, term.data, dims=data_dims, model=model)

    # Register parameter
    param_coords = coords | model.__bambi_attrs__["response_coords_reduced"]
    param_dims = tuple(param_coords)
    param_shape = tuple(len(coord) for coord in param_coords.values())

    # Makes sure arguments are of the shape implied by dims and their coords
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        dist(param_name, **kwargs, dims=param_dims)

    return model[data_name], model[param_name]
