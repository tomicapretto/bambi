import numpy as np
import pymc as pm

from bambi.backend.pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)

from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.types import Dims
from bambi.families.family import Family
from bambi.families.types import ResponseType


def build_response_term(term, parameters: dict, family: Family, model: pm.Model) -> None:
    if family.DATA_TYPE == ResponseType.BINARY:
        data = prepare_binary_data(term)
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        data = prepare_categorical_data(term)
    else:
        data = term.data

    dims = response_dims(family, model)
    distribution = get_distribution_from_likelihood(family.likelihood)

    transform_parameters = transforms_registry.get_transform_parameters(family)
    if transform_parameters:
        parameters = transform_parameters(parameters)

    if term.is_censored:
        observed = data[:, 0]
        censoring_code = data[:, 1]

        is_left_censored = censoring_code == -1
        is_right_censored = censoring_code == 1

        lower = np.where(is_left_censored, observed, -np.inf)
        upper = np.where(is_right_censored, observed, np.inf)
        dist = distribution.dist(**parameters)
        with model:
            pm.Censored(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)
    elif term.is_truncated:
        observed = data[:, 0]
        lower = data[:, 1]
        upper = data[:, 2]

        # Handle 'None' and scalars appropriately
        if np.all(lower == -np.inf):
            lower = None
        elif np.all(lower == lower[0]):
            lower = lower[0]

        if np.all(upper == np.inf):
            upper = None
        elif np.all(upper == upper[0]):
            upper = upper[0]

        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    elif term.is_constrained:
        # Handle constrained responses through truncated distributions
        observed = data[:, 0]
        lower = data[:, 1]
        upper = data[:, 2]

        # Handle 'None' and scalars appropriately
        if np.all(lower == -np.inf):
            lower = None
        elif np.all(lower == lower[0]):
            lower = lower[0]

        if np.all(upper == np.inf):
            upper = None
        elif np.all(upper == upper[0]):
            upper = upper[0]

        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    elif term.is_weighted:
        observed = data[:, 0]
        weights = data[:, 1]
        weighted_dist = make_weighted_distribution(distribution)

        with model:
            weighted_dist(term.label, weights, **parameters, observed=observed, dims=dims)
    else:

        transform_data = transforms_registry.get_transform_data(family)
        if transform_data:
            data_mapping = transform_data(data)
        else:
            data_mapping = {"observed": data}

        with model:
            # All of the other response kinds are not special and are thus handled the same way
            distribution(term.label, **parameters, **data_mapping, dims=dims)

    return None


def response_dims(family: Family, model: pm.Model) -> Dims:
    coords = model.__bambi_attrs__["response_coords_data"]

    response_is_indexed = family.DATA_TYPE in (
        ResponseType.BINARY,
        ResponseType.CATEGORICAL,
        ResponseType.ORDINAL,
    )
    if response_is_indexed:
        return tuple(coords)

    return tuple(coords | model.__bambi_attrs__["response_coords"])


def prepare_binary_data(term) -> np.ndarray:
    # Data is 2d when the user passes categorical response without specifying the reference level.
    # In that case, data is a one-hot encoded matrix. Otherwise it's a binary 1d array.
    if term.data.ndim == 1:
        return term.data
    idx = term.levels.index(term.reference)
    return term.data[:, idx]


def prepare_categorical_data(term) -> np.ndarray:
    # Data is a one-hot encoded matrix. PyMC needs a vector of indices of observed categories.
    return np.nonzero(term.data)[1]
