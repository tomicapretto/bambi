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

# NOTE: There is a ton of AI generated code here. It's a very bad mess.


def build_response_term(term, parameters: dict, family: Family, model: pm.Model) -> None:
    data = prepare_response_data(term, family)
    dims = get_response_dims(family, model)
    model.__bambi_attrs__["response_data"] = []

    distribution = get_distribution_from_likelihood(family.likelihood)

    transform_parameters = transforms_registry.get_transform_parameters(family)
    if transform_parameters:
        parameters = transform_parameters(parameters)

    if term.is_censored:
        observed = register_response_data(
            term, model, data[:, 0], dims, "observed", column=0, update_for_prediction=True
        )
        censoring_code = register_response_data(
            term,
            model,
            data[:, 1],
            dims,
            "censoring_code",
            column=1,
            update_for_prediction=True,
        )

        is_left_censored = pt.eq(censoring_code, -1)
        is_right_censored = pt.eq(censoring_code, 1)

        lower = pt.switch(is_left_censored, observed, -np.inf)
        upper = pt.switch(is_right_censored, observed, np.inf)
        dist = distribution.dist(**parameters)
        with model:
            pm.Censored(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)
    elif term.is_truncated:
        observed = register_response_data(term, model, data[:, 0], dims, "observed", column=0)
        lower = get_truncation_bound(term, model, data, dims, "lower")
        upper = get_truncation_bound(term, model, data, dims, "upper")
        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    elif term.is_constrained:
        # Handle constrained responses through truncated distributions
        observed = register_response_data(term, model, data[:, 0], dims, "observed", column=0)
        lower = get_truncation_bound(term, model, data, dims, "lower")
        upper = get_truncation_bound(term, model, data, dims, "upper")
        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(term.label, dist, lower=lower, upper=upper, observed=observed, dims=dims)

    elif term.is_weighted:
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
    else:
        if needs_trials_data(family) and data.ndim == 2:
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
            transform_data = transforms_registry.get_transform_data(family)
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


def get_response_data(term, family: Family, model: pm.Model, data, for_prediction: bool = False):
    values = {}
    full_data = None
    for info in model.__bambi_attrs__["response_data"]:
        if for_prediction and not info["update_for_prediction"]:
            continue

        if info["source"] is None:
            if full_data is None:
                full_data = prepare_response_data(term, family, term.eval_new_data(data))
            value = full_data if info["column"] is None else full_data[:, info["column"]]
        else:
            value = evaluate_call_arg(info["source"], data)

        values[info["name"]] = value

    return values


def prepare_response_data(term, family: Family, data: np.ndarray | None = None) -> np.ndarray:
    data = term.data if data is None else data
    if family.DATA_TYPE == ResponseType.BINARY:
        # Data is 2d when the user passes categorical response without specifying the reference
        # level. In that case, data is a one-hot encoded matrix. Otherwise it's a binary 1d array.
        if data.ndim == 1:
            return data
        idx = term.levels.index(term.reference)
        return data[:, idx]
    if family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        # Data is a one-hot encoded matrix. PyMC needs a vector of observed category indices.
        return np.nonzero(data)[1]
    return data


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


def get_truncation_bound(term, model: pm.Model, data: np.ndarray, dims: Dims, bound: str):
    column = 1 if bound == "lower" else 2
    no_bound_value = -np.inf if bound == "lower" else np.inf
    source = get_bound_source(term, bound)
    values = data[:, column]

    if source is None or not is_data_dependent(source):
        if np.all(values == no_bound_value):
            return None
        return as_scalar(values)

    return register_response_data(
        term,
        model,
        values,
        dims,
        bound,
        column=column,
        source=source,
        update_for_prediction=True,
    )


def as_scalar(values: np.ndarray):
    if np.all(values == values[0]):
        return values[0].item() if hasattr(values[0], "item") else values[0]
    return values


def get_bound_source(term, bound: str):
    if bound == "lower":
        return get_call_arg(term, 1, "lb")
    return get_call_arg(term, 2, "ub")


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


def evaluate_call_arg(value, data):
    env = None
    if isinstance(value, tuple):
        value, env = value
    if not hasattr(value, "eval"):
        return value
    result = value.eval(data, env)
    if hasattr(result, "eval"):
        return result.eval()
    return result


def get_response_data_dims(term, data: np.ndarray, dims: Dims, model: pm.Model) -> Dims:
    if data.ndim <= len(dims):
        return dims

    extra_dims = tuple(f"{term.label}_data_dim_{i}" for i in range(data.ndim - len(dims)))
    extra_coords = {
        dim: range(data.shape[len(dims) + index]) for index, dim in enumerate(extra_dims)
    }
    model.add_coords(extra_coords)
    return tuple(dims) + extra_dims


def get_response_dims(family: Family, model: pm.Model) -> Dims:
    coords = model.__bambi_attrs__["response_coords_data"]

    response_is_indexed = family.DATA_TYPE in (
        ResponseType.BINARY,
        ResponseType.CATEGORICAL,
        ResponseType.ORDINAL,
    )
    if response_is_indexed:
        return tuple(coords)

    return tuple(coords | model.__bambi_attrs__["response_coords"])
