import numpy as np
import pymc as pm

from bambi.backend.pymc.coords import coords_from_common
from bambi.backend.pymc.utils import get_distribution_from_prior


def shape_data(data, coords):
    if not coords:
        return data

    shape = tuple(len(coord) for coord in coords.values())
    size = np.prod(shape)
    data = np.asarray(data)

    if data.ndim == len(shape) + 1 and data.shape[1:] == shape:
        return data

    if data.ndim == 1:
        if size != 1:
            raise ValueError("Cannot reshape one-dimensional common term data to multiple levels.")
        return data[:, np.newaxis]

    if data.ndim == 2 and data.shape[1] == size:
        return data.reshape((data.shape[0], *shape))

    raise ValueError("Common term data shape does not match its coordinates.")


def flatten_data(data, coords):
    if not coords:
        return data
    return data.reshape((data.shape[0], -1))


def flatten_param(param, term_coords, response_coords):
    if not term_coords:
        return param

    response_shape = tuple(len(coord) for coord in response_coords.values())
    if response_shape:
        return param.reshape((-1, *response_shape))
    return param.reshape((-1,))


def shape_prior_arg(value, shape):
    value = np.asarray(value)
    if value.shape == shape:
        return value
    if value.size == np.prod(shape):
        return value.reshape(shape)
    return np.broadcast_to(value, shape)


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
        data_dims = ("__obs__", *coords)
        data = shape_data(term.data, coords)
        pm.Data(data_name, data, dims=data_dims, model=model)

    # Register parameter
    response_coords = model.__bambi_attrs__["response_coords_reduced"]
    param_coords = coords | response_coords
    param_dims = tuple(param_coords)
    param_shape = tuple(len(coord) for coord in param_coords.values())

    # Makes sure arguments are of the shape implied by dims and their coords
    kwargs = {name: shape_prior_arg(value, param_shape) for name, value in term.prior.args.items()}
    dist = get_distribution_from_prior(term.prior)

    with model:
        dist(param_name, **kwargs, dims=param_dims)

    data = flatten_data(model[data_name], coords)
    param = flatten_param(model[param_name], coords, response_coords)
    return data, param
