import numpy as np
import pymc as pm

from bambi.backend.new_pymc.coords import coords_from_common
from bambi.backend.new_pymc.utils import get_distribution_from_prior


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
    data_name = f"{term.name}_data"
    param_name = term.name
    coords = coords_from_common(term)

    # Register coords
    if data_name not in model or param_name not in model:
        model.add_coords(coords)

    # Register data
    if data_name not in model:
        # Reshapes to 2D if not already
        data_dims = ("__obs__", *coords)
        data_shape = (term.data.shape[0], -1)
        pm.Data(data_name, np.reshape(term.data, data_shape), dims=data_dims, model=model)

    # Register parameter
    output_coords = {
        k: v for k, v in model.__bambi_attrs__["output_coords"].items() if k != "__obs__"
    }
    param_coords = coords | output_coords
    param_dims = tuple(param_coords)
    param_shape = tuple(len(coord) for coord in param_coords.values())

    # Makes sure arguments are of the shape implied by dims and their coords
    kwargs = {name: np.broadcast_to(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)
    dist(param_name, **kwargs, dims=param_dims, model=model)

    return model[data_name], model[param_name]
