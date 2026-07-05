import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.coords import coords_from_common
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.types import CoefSpec, Constraint, Coords


def shape_data(data: np.ndarray, coords: Coords) -> np.ndarray:
    if not coords:
        # Without term coords, PyMC data is registered only over observations.
        # Single-column design matrices therefore represent scalar terms and become vectors.
        if data.ndim == 2 and data.shape[1] == 1:
            return data[:, 0]
        if data.ndim > 1:
            raise ValueError("Common term data without coordinates must be one-dimensional.")
        return data

    # Coords describe all non-observation dimensions for the term.
    # Formulae usually gives common terms as flat design-matrix columns,
    # while PyMC data is registered with named dimensions.
    # This restores the named coordinate shape.
    shape = tuple(len(coord) for coord in coords.values())
    size = np.prod(shape)

    # Data may already be shaped as (__obs__, *coords), e.g. after a model update.
    if data.ndim == len(shape) + 1 and data.shape[1:] == shape:
        return data

    # A vector can only be expanded when the coords imply a single column.
    if data.ndim == 1:
        if size != 1:
            raise ValueError("Cannot reshape one-dimensional common term data to multiple levels.")
        return data[:, np.newaxis]

    # Convert flat design-matrix columns back into the named coordinate dimensions.
    if data.ndim == 2 and data.shape[1] == size:
        return data.reshape((data.shape[0], *shape))

    raise ValueError("Common term data shape does not match its coordinates.")


def flatten_data(data: pt.Variable, coords: Coords) -> pt.Variable:
    if not coords:
        return data
    # The linear predictor is computed with dot(data, params),
    # so named term dimensions are flattened back into design-matrix columns.
    return data.reshape((data.shape[0], -1))


def flatten_param(param: pt.Variable, term_coords: Coords, response_coords: Coords) -> pt.Variable:
    if not term_coords:
        return param

    # Match the flattened term data: dimensions collapse into one coefficient axis,
    # while response dimensions stay separate.
    response_shape = tuple(len(coord) for coord in response_coords.values())
    if response_shape:
        return param.reshape((-1, *response_shape))
    return param.reshape((-1,))


def shape_prior_arg(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if value.shape == shape:
        return value

    if value.size == np.prod(shape):
        # Flat prior arguments are interpreted in the same coordinate order as the parameter.
        return value.reshape(shape)

    if value.shape == shape[: value.ndim]:
        # Term-shaped prior arguments broadcast across response dimensions.
        value = value.reshape((*value.shape, *(1 for _ in shape[value.ndim :])))

    return np.broadcast_to(value, shape)


def build_common_term(
    term, coef_spec: CoefSpec, model: pm.Model
) -> tuple[pt.Variable, pt.Variable]:
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
    response_coords = {}
    if coef_spec.ndim > 0:
        if coef_spec.constraint == Constraint.REFERENCE:
            response_coords = model.__bambi_attrs__["response_coords_reduced"]
        else:
            response_coords = model.__bambi_attrs__["response_coords"]

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
