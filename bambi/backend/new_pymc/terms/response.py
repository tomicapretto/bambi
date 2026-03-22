import numpy as np
import pymc as pm

from bambi.backend.new_pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)


def build_response_term(term, parameters, family, model):
    kwargs = parameters | {
        "observed": term.data,
        "dims": tuple(model.__bambi_attrs__["response_coords"]),
    }
    if hasattr(family, "transform_response_kwargs"):
        kwargs = family.transform_response_kwargs(kwargs)

    distribution = get_distribution_from_likelihood(family.likelihood)

    if term.is_censored:
        dims = kwargs.pop("dims", None)
        data_matrix = kwargs.pop("observed")

        # Get values of the response variable
        observed = np.squeeze(data_matrix[:, 0])

        # Get censoring codes
        censoring_code = np.squeeze(data_matrix[:, 1])

        is_left_censored = censoring_code == -1
        is_right_censored = censoring_code == 1

        lower = np.where(is_left_censored, observed, -np.inf)
        upper = np.where(is_right_censored, observed, np.inf)
        dist = distribution.dist(**kwargs)
        with model:
            pm.Censored(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)
    elif term.is_truncated:
        dims = kwargs.pop("dims", None)
        data_matrix = kwargs.pop("observed")

        # Get values of the response variable
        observed = np.squeeze(data_matrix[:, 0])

        # Get truncation values
        lower = np.squeeze(data_matrix[:, 1])
        upper = np.squeeze(data_matrix[:, 2])

        # Handle 'None' and scalars appropriately
        if np.all(lower == -np.inf):
            lower = None
        elif np.all(lower == lower[0]):
            lower = lower[0]

        if np.all(upper == np.inf):
            upper = None
        elif np.all(upper == upper[0]):
            upper = upper[0]

        dist = distribution.dist(**kwargs)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    elif term.is_constrained:
        # Handle constrained responses (through truncated distributions)
        dims = kwargs.pop("dims", None)
        data_matrix = kwargs.pop("observed")

        # Get values of the response variable
        observed = np.squeeze(data_matrix[:, 0])

        # Get truncation values
        lower = np.squeeze(data_matrix[:, 1])
        upper = np.squeeze(data_matrix[:, 2])

        # Handle 'None' and scalars appropriately
        if np.all(lower == -np.inf):
            lower = None
        elif np.all(lower == lower[0]):
            lower = lower[0]

        if np.all(upper == np.inf):
            upper = None
        elif np.all(upper == upper[0]):
            upper = upper[0]

        dist = distribution.dist(**kwargs)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    # Handle weighted responses
    elif term.is_weighted:
        dims = kwargs.pop("dims", None)
        data_matrix = kwargs.pop("observed")

        # Get values of the response variable
        observed = np.squeeze(data_matrix[:, 0])

        # Get weights
        weights = np.squeeze(data_matrix[:, 1])

        # Get a weighted version of the response distribution
        weighted_dist = make_weighted_distribution(distribution)

        with model:
            weighted_dist(term.label, weights, **kwargs, observed=observed, dims=dims)
    else:
        # All of the other response kinds are not special and are thus handled the same way
        distribution(term.label, **kwargs, model=model)

    return None
