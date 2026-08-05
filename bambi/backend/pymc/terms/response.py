import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)
from bambi.backend.pymc.transform import transforms_registry
from bambi.families.family import Family
from bambi.families.types import ResponseType
from bambi.terms.response import ResponseTerm


# NOTE: There is a ton of AI generated code here. It's a very bad mess.
def build_response_term(
    term: ResponseTerm, parameters: dict, family: Family, model: pm.Model
) -> None:
    distribution = get_distribution_from_likelihood(family.likelihood)
    data = term.data

    if family.DATA_TYPE == ResponseType.BINARY and data.ndim > 1:
        # Data is 2D when the user passes a categoric response without setting the reference level.
        # In that case, data is a one-hot encoded matrix and we select the corresponding column.
        # Otherwise data is already a 1D binary array and we don't need to do anything.
        index = term.levels.index(term.reference)
        data = data[:, index]
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        # Data is a one-hot encoded matrix. PyMC needs a vector of observed category indices.
        data = np.nonzero(data)[1]

    # All families get coordinates for observation indexes.
    # Multidimensional models also get additional coords, if available.
    dims = tuple(model.__bambi_attrs__["response_coords_data"])
    if family.RESPONSE_NDIM > 0:
        dims = dims + tuple(model.__bambi_attrs__["response_coords"])

    transform_parameters = transforms_registry.get_parameter_transform(family)
    if transform_parameters is not None:
        parameters = transform_parameters(parameters)

    # NOTE: Does it make sense to use term.label + "_data"?
    #       Shouldn't I use the variable name instead?
    if term.is_censored:
        # NOTE: Graph intervention for predictions
        observed = pm.Data(term.label + "_data", data[:, 0], dims=dims, model=model)
        censoring_code = pm.Data(term.label + "_status", data[:, 1], dims=dims, model=model)

        # When there's no left or right censoring, avoid pytensor constructs.
        # Left censoring
        if not any(data[:, 1] == -1):
            lower = -np.inf
        else:
            is_left_censored = pt.eq(censoring_code, -1)
            lower = pt.switch(is_left_censored, observed, -np.inf)

        # Right censoring
        if not any(data[:, 1] == 1):
            upper = np.inf
        else:
            is_right_censored = pt.eq(censoring_code, 1)
            upper = pt.switch(is_right_censored, observed, np.inf)

        dist = distribution.dist(**parameters)

        with model:
            pm.Censored(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

        return None

    if term.is_truncated or term.is_constrained:
        # NOTE: Predictions: truncated requires us to remove Truncated, constrained does not.
        lower_data = data[:, 1]
        upper_data = data[:, 2]
        observed = pm.Data(term.label + "_data", data[:, 0], dims=dims, model=model)

        if all(lower_data == -np.inf):
            lower = None
        elif np.all(lower_data == lower_data[0]):
            # NOTE: They could all be equal even when we pass a variable instead of a literal.
            lower = lower_data[0]
        else:
            lower = pm.Data(term.label + "_lb", lower_data, dims=dims, model=model)

        if all(upper_data == np.inf):
            upper = None
        elif np.all(upper_data == upper_data[0]):
            # NOTE: They could all be equal even when we pass a variable instead of a literal.
            upper = upper_data[0]
        else:
            upper = pm.Data(term.label + "_ub", upper_data, dims=dims, model=model)

        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

        return None

    if term.is_weighted:
        # TODO: Do we need to intervene for predictions?
        #       This weighting only matters in the likelihood, but is not related to predictions.
        observed = pm.Data(term.label + "_data", data[:, 0], dims=dims, model=model)
        weights = pm.Data(term.label + "_weights", data[:, 1], dims=dims, model=model)
        weighted_dist = make_weighted_distribution(distribution)

        with model:
            weighted_dist(term.label, weights, **parameters, observed=observed, dims=dims)

        return None

    transform_data = transforms_registry.get_data_transform(family)
    if transform_data is not None:
        data_mapping = transform_data(data)
    else:
        data_mapping = {"observed": data}

    # TODO: Can we do better at naming for function calls?
    #       In general, we could inspect on which variable names calls depend on.
    #       That is not a guarantee the function call returns that variable as a column.
    #       For example, p(y, n) does that, but it can be an exception.
    #       p(y, trials). Which name to use for each variable?
    #       c(y1, y2, y3, y4) -> this is different, the concatenation actualle matters
    #       log(y) ... can't avoid log(y)_data
    # NOTE: Why don't we use 'clean' names for things we know (p, c, etc.) and go back
    #       to defaults in things we don't know, such as a regular function call?

    with model:
        distribution(term.label, **parameters, **data_mapping, dims=dims)

    return None
