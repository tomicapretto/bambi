import inspect
from typing import Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from formulae.terms.call_utils import CallVarsExtractor
from formulae.terms.call_resolver import get_function_from_module

from bambi.backend.pymc.utils import (
    make_weighted_distribution,
    get_distribution_from_likelihood,
)
from bambi.backend.pymc.transform import transforms_registry
from bambi.families.family import Family
from bambi.families.types import ResponseType
from bambi.terms.response import ResponseTerm


def build_response_term(
    term: ResponseTerm, parameters: dict, family: Family, model: pm.Model
) -> None:
    distribution = get_distribution_from_likelihood(family.likelihood)

    # All families get coordinates for observation indexes.
    # Multidimensional models also get additional coords, if available.
    dims = tuple(model.__bambi_attrs__["response_coords_data"])
    if family.RESPONSE_NDIM > 0:
        dims = dims + tuple(model.__bambi_attrs__["response_coords"])

    transform_parameters = transforms_registry.get_parameter_transform(family)
    if transform_parameters is not None:
        parameters = transform_parameters(parameters)

    if term.is_censored:
        # NOTE: Graph intervention for predictions
        # NOTE: Still need to handle interval censoring.
        # NOTE: Statuses could be more efficient (in some cases) if we allowed for scalars.
        #       For now, statuses are vectors of the same length as observed data.
        var_names = list(_get_call_bound_arguments(term))
        observed, status = term.data[:, 0], term.data[:, 1]
        observed_data = pm.Data(var_names[0] + "_data", observed, dims=dims, model=model)
        status_data = pm.Data(var_names[1] + "_data", status, dims=dims, model=model)

        # Avoid PyTensor constructs when there's no such a censoring type.
        # Left censoring
        if not any(status == -1):
            lower = -np.inf
        else:
            is_left_censored = pt.eq(status_data, -1)
            lower = pt.switch(is_left_censored, observed_data, -np.inf)

        # Right censoring
        if not any(status == 1):
            upper = np.inf
        else:
            is_right_censored = pt.eq(status_data, 1)
            upper = pt.switch(is_right_censored, observed_data, np.inf)

        dist = distribution.dist(**parameters)
        with model:
            pm.Censored(
                term.label, dist, lower=lower, upper=upper, observed=observed_data, dims=dims
            )
        return None

    if term.is_truncated or term.is_constrained:
        observed, lower, upper = term.data[:, 0], term.data[:, 1], term.data[:, 2]
        call_args = _get_call_bound_arguments(term)
        value_name = call_args["x"]
        observed_data = pm.Data(value_name + "_data", observed, dims=dims, model=model)

        if "lb" in call_args:
            if call_args["lb"] == "":
                # A literal, all observations share the same lower bound.
                lower_data = lower[0].item()
            else:
                # A variable name, lower bound is a vector of the same length as observed data.
                lower_name = call_args["lb"]
                lower_data = pm.Data(lower_name + "_data", lower, dims=dims, model=model)
        else:
            lower_data = None

        if "ub" in call_args:
            if call_args["ub"] == "":
                # A literal, all observations share the same upper bound.
                upper_data = upper[0].item()
            else:
                # A variable name, upper bound is a vector of the same length as observed data.
                upper_name = call_args["ub"]
                upper_data = pm.Data(upper_name + "_data", upper, dims=dims, model=model)
        else:
            upper_data = None

        dist = distribution.dist(**parameters)
        with model:
            pm.Truncated(
                term.label,
                dist,
                lower=lower_data,
                upper=upper_data,
                observed=observed_data,
                dims=dims,
            )
        return None

    if term.is_weighted:
        observed, weights = term.data[:, 0], term.data[:, 1]
        call_args = _get_call_bound_arguments(term)

        value_name = call_args["x"]
        observed_data = pm.Data(value_name + "_data", observed, dims=dims, model=model)

        if call_args["weights"] == "":
            # A literal, all observations share the same weight.
            weights_data = weights[0].item()
        else:
            weights_name = call_args["weights"]
            weights_data = pm.Data(weights_name + "_data", weights, dims=dims, model=model)

        weighted_dist = make_weighted_distribution(distribution)

        with model:
            weighted_dist(term.label, weights_data, **parameters, observed=observed_data, dims=dims)
        return None

    if term.is_binomial:
        successes, trials = term.data[:, 0], term.data[:, 1]
        call_args = _get_call_bound_arguments(term)

        successes_name = call_args["successes"]
        successes_data = pm.Data(successes_name + "_data", successes, dims=dims, model=model)

        if call_args["trials"] == "":
            # A literal, all observations share the same number of trials.
            trials_data = trials[0].item()
        else:
            trials_name = call_args["trials"]
            trials_data = pm.Data(trials_name + "_data", trials, dims=dims, model=model)

        with model:
            distribution(
                term.label, **parameters, observed=successes_data, n=trials_data, dims=dims
            )

        return None

    data = term.data
    if family.DATA_TYPE == ResponseType.BINARY and data.ndim > 1:
        # In a binary response model, when the user uses a categoric response without setting the
        # reference level, the data will be a 2D one-hot encoded matrix.
        # In that case, we select the corresponding column for the reference level.
        # Otherwise, the data is already a 1D binary array and no further action is needed.
        index = term.levels.index(term.reference)
        data = data[:, index]
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        # In categorical and ordinal response models, the data is a 2D one-hot encoded matrix,
        # but PyMC needs a vector of observed category indices.
        data = np.nonzero(data)[1]

    transform_data = transforms_registry.get_data_transform(family)
    if transform_data is not None:
        data_mapping = transform_data(data)
    else:
        data_mapping = {"observed": data}

    data_vars = {}
    for name, value in data_mapping.items():
        if name == "observed":
            label = term.label + "_data"
        else:
            label = name + "_data"

        data_vars[name] = pm.Data(label, value, dims=dims, model=model)

    with model:
        distribution(term.label, **parameters, **data_vars, dims=dims)

    return None


Purpose = Literal["prediction", "log_likelihood"]


def build_new_response_data(
    term: ResponseTerm, data: pd.DataFrame, family: Family, purpose: Purpose
):
    if term.is_censored:
        return _build_new_censored_data(term, data, purpose)

    if term.is_truncated:
        return _build_new_truncated_data(term, data, purpose)

    if term.is_constrained:
        return _build_new_constrained_data(term, data, purpose)

    if term.is_weighted:
        return _build_new_weighted_data(term, data, purpose)

    if term.is_binomial:
        return _build_new_binomial_data(term, data, purpose)

    return _build_new_generic_data(term, data, family, purpose)


def _build_new_censored_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    status_name = call_args["status"]
    if purpose == "prediction":
        # If response term variables are available, compute conditional predictions when possible,
        # such as p(Y | Y > t) when the observation is right-censored.
        # For non-censored observations, it generates predictions for the latent variable.
        # If response term variables are not available, generate predictions for the latent variable
        # in all cases.
        if value_name not in data.columns or status_name not in data.columns:
            # Latent variable predictions
            # NOTE: Use values compatible with the model
            return {
                value_name + "_data": np.zeros(data.shape[0]),
                status_name + "_data": np.zeros(data.shape[0], dtype=int),
            }

        # Conditional predictions
        response_data = term.eval_new_data(data)
        value, status = response_data[:, 0], response_data[:, 1]
        return {value_name + "_data": value, status_name + "_data": status}

    if purpose == "log_likelihood":
        # If there is no status, we assume the user wants prediction the latent variable.
        # If there is a status, the status controls if it's conditional or latent.
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")

        if status_name not in data.columns:
            data = data[[value_name]].copy()
            data[status_name] = "none"

        response_data = term.eval_new_data(data)
        value, status = response_data[:, 0], response_data[:, 1]
        return {value_name + "_data": value, status_name + "_data": status}

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_new_truncated_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    lower_name = call_args.get("lb", "")
    upper_name = call_args.get("ub", "")

    if purpose == "prediction":
        # Predictions are truncated when lb or ub are either literals or
        # when they are variable names available in 'data'.
        include_lower = lower_name and lower_name in data.columns
        include_upper = upper_name and upper_name in data.columns

        output = {value_name: np.zeros(data.shape[0])}
        if include_lower or include_upper:
            if include_lower:
                output[lower_name] = data[lower_name]
            else:
                output[lower_name] = np.zeros(data.shape[0])

            if include_upper:
                output[upper_name] = data[upper_name]
            else:
                output[upper_name] = np.zeros(data.shape[0])

            df_dummy = pd.DataFrame(output)
            response_data = term.eval_new_data(df_dummy)
            value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]
            output = {value_name + "_data": value}

            if include_lower:
                output[lower_name + "_data"] = lower
            if include_upper:
                output[upper_name + "_data"] = upper

            return output
        return output

    if purpose == "log_likelihood":
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")

        include_lower = lower_name and lower_name in data.columns
        include_upper = upper_name and upper_name in data.columns

        df_dummy = data[[value_name]].copy()
        if include_lower:
            df_dummy[lower_name] = data[lower_name]
        if lower_name and not include_lower:
            df_dummy[lower_name] = np.zeros(data.shape[0])

        if include_upper:
            df_dummy[upper_name] = data[upper_name]
        if upper_name and not include_upper:
            df_dummy[upper_name] = np.zeros(data.shape[0])

        response_data = term.eval_new_data(df_dummy)
        value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]
        output = {value_name + "_data": value}

        if include_lower:
            output[lower_name + "_data"] = lower
        if include_upper:
            output[upper_name + "_data"] = upper

        return output

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_new_constrained_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    lower_name = call_args.get("lb", "")
    upper_name = call_args.get("ub", "")
    bound_names = [name for name in (lower_name, upper_name) if name]

    if purpose == "prediction":
        missing_var_names = [name for name in bound_names if name not in data.columns]
        if missing_var_names:
            raise ValueError(
                f"Response term variable '{missing_var_names[0]}' must be present in the data."
            )

        df_dummy = pd.DataFrame({value_name: np.zeros(data.shape[0])})
        for name in bound_names:
            df_dummy[name] = data[name]

        response_data = term.eval_new_data(df_dummy)
        value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]
        output = {value_name + "_data": value}

        if lower_name:
            output[lower_name + "_data"] = lower
        if upper_name:
            output[upper_name + "_data"] = upper

        return output

    if purpose == "log_likelihood":
        var_names = [value_name] + bound_names
        missing_var_names = [name for name in var_names if name not in data.columns]
        if missing_var_names:
            raise ValueError(
                f"Response term variable '{missing_var_names[0]}' must be present in the data."
            )

        df_dummy = data[[value_name]].copy()
        for name in bound_names:
            df_dummy[name] = data[name]

        response_data = term.eval_new_data(df_dummy)
        value, lower, upper = response_data[:, 0], response_data[:, 1], response_data[:, 2]
        output = {value_name + "_data": value}

        if lower_name:
            output[lower_name + "_data"] = lower
        if upper_name:
            output[upper_name + "_data"] = upper

        return output

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_new_weighted_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    value_name = call_args["x"]
    weights_name = call_args.get("weights", "")

    if purpose == "prediction":
        output = {value_name + "_data": np.zeros(data.shape[0])}

        if weights_name:
            output[weights_name + "_data"] = np.ones(data.shape[0])

        return output

    if purpose == "log_likelihood":
        if value_name not in data.columns:
            raise ValueError(f"Response term variable '{value_name}' must be present in the data.")

        df_dummy = data[[value_name]].copy()
        if weights_name:
            if weights_name in data.columns:
                df_dummy[weights_name] = data[weights_name]
            else:
                df_dummy[weights_name] = np.ones(data.shape[0])

        response_data = term.eval_new_data(df_dummy)
        value, weights = response_data[:, 0], response_data[:, 1]
        output = {value_name + "_data": value}

        if weights_name:
            output[weights_name + "_data"] = weights

        return output

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_new_binomial_data(term: ResponseTerm, data: pd.DataFrame, purpose: Purpose):
    call_args = _get_call_bound_arguments(term)
    successes_name = call_args["successes"]
    trials_name = call_args.get("trials", "")
    include_trials = trials_name and trials_name in data.columns

    if purpose == "prediction":
        output = {successes_name + "_data": np.zeros(data.shape[0])}

        if include_trials:
            df_dummy = pd.DataFrame(
                {
                    successes_name: np.zeros(data.shape[0]),
                    trials_name: data[trials_name],
                }
            )
            response_data = term.eval_new_data(df_dummy)
            output[trials_name + "_data"] = response_data[:, 1]

        return output

    if purpose == "log_likelihood":
        if successes_name not in data.columns:
            raise ValueError(
                f"Response term variable '{successes_name}' must be present in the data."
            )

        df_dummy = data[[successes_name]].copy()
        if include_trials:
            df_dummy[trials_name] = data[trials_name]

        response_data = term.eval_new_data(df_dummy)
        successes, trials = response_data[:, 0], response_data[:, 1]
        output = {successes_name + "_data": successes}

        if include_trials:
            output[trials_name + "_data"] = trials

        return output

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_new_generic_data(
    term: ResponseTerm, data: pd.DataFrame, family: Family, purpose: Purpose
):
    var_names = list(term.term.var_names)

    if purpose == "prediction":
        df_dummy = pd.DataFrame(index=data.index)
        for name in var_names:
            if name in data.columns:
                df_dummy[name] = data[name]
            else:
                df_dummy[name] = np.zeros(data.shape[0])

        response_data = term.eval_new_data(df_dummy)
        data_mapping = _build_response_data_mapping(term, response_data, family)
        output = {}

        for name, value in data_mapping.items():
            if name == "observed":
                output[term.label + "_data"] = np.zeros_like(value)
            elif name in data.columns:
                output[name + "_data"] = data[name]
            else:
                output[name + "_data"] = value

        return output

    if purpose == "log_likelihood":
        missing_var_names = [name for name in var_names if name not in data.columns]
        if missing_var_names:
            raise ValueError(
                f"Response term variable '{missing_var_names[0]}' must be present in the data."
            )

        response_data = term.eval_new_data(data[var_names])
        data_mapping = _build_response_data_mapping(term, response_data, family)
        output = {}

        for name, value in data_mapping.items():
            if name == "observed":
                output[term.label + "_data"] = value
            else:
                output[name + "_data"] = value

        return output

    raise ValueError(f"Unsupported purpose: {purpose}")


def _build_response_data_mapping(term: ResponseTerm, data: np.ndarray, family: Family):
    if family.DATA_TYPE == ResponseType.BINARY and data.ndim > 1:
        index = term.levels.index(term.reference)
        data = data[:, index]
    elif family.DATA_TYPE in (ResponseType.CATEGORICAL, ResponseType.ORDINAL):
        data = np.nonzero(data)[1]

    transform_data = transforms_registry.get_data_transform(family)
    if transform_data is not None:
        return transform_data(data)
    return {"observed": data}


def _get_call_bound_arguments(term: ResponseTerm) -> dict:
    component = term.components[0]
    function = get_function_from_module(component.call.callee, component.env)
    bound = inspect.signature(function).bind(*component.call.args, **component.call.kwargs)
    parameters = list(dict(bound.arguments))
    arguments = CallVarsExtractor(component.call).get()
    return dict(zip(parameters, arguments))
