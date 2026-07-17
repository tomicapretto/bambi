import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)
from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.types import Dims
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
    parameters = transform_parameters(parameters)

    if term.is_censored:
        # NOTE: For predictions, I think we need to intervene the graph when the 'y' values
        #       are given. Recall VV project.
        observed = pm.Data(term.label + "_data", data[:, 0], dims=dims, model=model)
        censoring_code = pm.Data(term.label + "_status_data", data[:, 1], dims=dims, model=model)

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
            lower = pm.Data(term.label + "_lb_data", lower_data, dims=dims, model=model)

        if all(upper_data == np.inf):
            upper = None
        elif np.all(upper_data == upper_data[0]):
            # NOTE: They could all be equal even when we pass a variable instead of a literal.
            upper = upper_data[0]
        else:
            upper = pm.Data(term.label + "_ub_data", upper_data, dims=dims, model=model)

        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

        return None

    if term.is_weighted:
        # TODO: Do we need to intervene for predictions?
        #       This weighting only matters in the likelihood, but is not related to predictions.
        observed = register_response_data(term, model, data[:, 0], dims, "observed", column=0)
        weights = register_response_data(
            term,
            model,
            data[:, 1],
            dims,
            "weights",
            column=1,
            source=get_call_arg(term, 1),
            update_for_prediction=True,
        )
        weighted_dist = make_weighted_distribution(distribution)

        with model:
            weighted_dist(term.label, weights, **parameters, observed=observed, dims=dims)

        return None

    if needs_trials_data(family) and data.ndim == 2:
        # TODO: 'needs' trials is too narrow. Also, I think the transform can/should handle this.
        observed = register_response_data(term, model, data[:, 0], dims, "observed", column=0)
        trials = get_trials_data(term, model, data, dims)
        data_mapping = {"observed": observed, "n": trials}
    else:
        data_dims = get_response_data_dims(term, data, dims, model)
        data = register_response_data(
            term,
            model,
            data,
            data_dims,
            "observed",
            update_for_prediction=needs_response_data_for_prediction(family),
        )

        transform_data = transforms_registry.get_data_transform(family)
        if transform_data:
            data_mapping = transform_data(data)
        else:
            data_mapping = {"observed": data}

        with model:
            # All of the other response kinds are not special and are thus handled the same way
            distribution(term.label, **parameters, **data_mapping, dims=dims)

    return None


def register_response_data(
    term,
    model: pm.Model,
    data: np.ndarray,
    dims: Dims,
    role: str,
    column: int | None = None,
    source=None,
    update_for_prediction: bool = False,
):
    data_name = get_response_data_name(term, role)
    data = pm.Data(data_name, data, dims=dims, model=model)

    if role == "observed":
        model.__bambi_attrs__["response_data_name"] = data_name
        model.__bambi_attrs__["response_data_dims"] = dims

    model.__bambi_attrs__["response_data"].append(
        {
            "name": data_name,
            "role": role,
            "column": column,
            "source": source,
            "update_for_prediction": update_for_prediction,
        }
    )
    return data


def get_response_data_name(term, role: str) -> str:
    if role == "observed":
        return f"{term.label}_data"
    return f"{term.label}_{role}_data"


def needs_trials_data(family: Family) -> bool:
    return family.likelihood.name in ("Binomial", "BetaBinomial", "ZeroInflatedBinomial")


def needs_response_data_for_prediction(family: Family) -> bool:
    return family.likelihood.name in ("Multinomial", "DirichletMultinomial")


def get_trials_data(term, model: pm.Model, data: np.ndarray, dims: Dims):
    source = get_call_arg(term, 1)
    trials = data[:, 1]
    if source is not None and is_data_dependent(source):
        return register_response_data(
            term,
            model,
            trials,
            dims,
            "n",
            column=1,
            source=source,
            update_for_prediction=True,
        )
    return as_scalar(trials)


def as_scalar(values: np.ndarray):
    if np.all(values == values[0]):
        return values[0].item() if hasattr(values[0], "item") else values[0]
    return values


def get_call_arg(term, position: int, keyword: str | None = None):
    if len(term.components) != 1:
        return None

    component = term.components[0]
    if not hasattr(component, "call"):
        return None

    if len(component.call.args) > position:
        return component.call.args[position], component.env

    if keyword is not None:
        value = component.call.kwargs.get(keyword)
        if value is not None:
            return value, component.env

    return None


def is_data_dependent(value) -> bool:
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return False
    if getattr(value, "name", None) is not None:
        return True
    if hasattr(value, "args"):
        return any(is_data_dependent(arg) for arg in value.args)
    return False


def get_response_data_dims(term, data: np.ndarray, dims: Dims, model: pm.Model) -> Dims:
    if data.ndim <= len(dims):
        return dims

    extra_dims = tuple(f"{term.label}_data_dim_{i}" for i in range(data.ndim - len(dims)))
    extra_coords = {
        dim: range(data.shape[len(dims) + index]) for index, dim in enumerate(extra_dims)
    }
    model.add_coords(extra_coords)
    return tuple(dims) + extra_dims
