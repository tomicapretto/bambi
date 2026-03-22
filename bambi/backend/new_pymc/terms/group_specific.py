import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.priors import Prior
from bambi.backend.new_pymc.coords import coords_from_group_specific
from bambi.backend.new_pymc.utils import get_distribution_from_prior

# Data will be either a matrix or a vector.
# If it is a matrix, it can be
# - (__obs__, *dims_factor)
# - (__obs__, *(expr_factor + dims_factor))


def build_group_specific_term_dot(term, model):
    # NOTE: Can we assume data_name is unique?
    data_name = f"{term.label}_data"
    param_name = term.label

    coords_expr, coords_factor = coords_from_group_specific(term)
    coords = coords_factor | coords_expr
    dims_expr = tuple(coords_expr)
    dims_factor = tuple(coords_factor)

    # Register coords
    # Data is not checked as no coords are registered for it
    if param_name not in model:
        model.add_coords(coords)

    # Register data (sparse matrix)
    data = term.data
    data_dims = ("__obs__", f"{term.label}_col")
    model.add_coords({data_dims[1]: np.arange(data.shape[1])})
    pm.Data(data_name, data, dims=data_dims, model=model)

    # Register parameter
    dims_output = tuple(model.__bambi_attrs__["response_coords"])[1:]
    param_rv = build_distribution(
        prior=term.prior,
        label=param_name,
        dims_expr=dims_expr,
        dims_factor=dims_factor,
        dims_output=dims_output,
        noncentered=term.noncentered,
        model=model,
    )

    # If response is multivariate: (q, K)
    # If response is univariate:   (q, )
    if dims_output:
        param_rv = param_rv.reshape(-1, param_rv.shape[-1])
    else:
        param_rv = param_rv.flatten()

    return model[data_name], param_rv


def build_group_specific_term_idx(term, model):
    data_value_name = f"{term.label}_data"
    data_idx_name = f"{term.label}_idx"
    param_name = term.label

    coords = term.coords.copy()
    if len(coords) == 1:
        (dims_factor,) = tuple(coords)
        dims_expr = tuple()
    elif len(coords) == 2:
        # Get strings, need tuples
        dims_factor, dims_expr = coords
        dims_factor = (dims_factor,)
        dims_expr = (dims_expr,)
    else:
        raise ValueError("no no!")

    # Register coords
    # Data is not checked as no coords are registered for it
    if param_name not in model:
        model.add_coords(coords)

    # Register data, predictor
    # TODO: Add dims for second dim and coords, if needed
    predictor_dims = ("__obs__",) + dims_expr
    predictor_data = pm.Data(data_value_name, term.predictor, dims=predictor_dims, model=model)

    # Register data, group index (which index of parameter to select from)
    group_idx_data = pm.Data(data_idx_name, term.group_index, dims=("__obs__",), model=model)

    # Register parameter
    dims_output = tuple(model.__bambi_attrs__["response_coords"])[1:]
    param_rv = build_distribution(
        prior=term.prior,
        label=param_name,
        dims_factor=dims_factor,
        dims_expr=dims_expr,
        dims_output=dims_output,
        noncentered=term.noncentered,
        model=model,
    )

    if dims_output:
        # (n, )    -> (n, 1)
        # (n, q_j) -> (n, q_j, 1)
        predictor_data = predictor_data[:, np.newaxis]

    # (n, ) * (n, )             -> (n, )
    # (n, q_j) * (n, q_j)       -> (n, q_j)
    # (n, K) * (n, 1)           -> (n, K)
    # (n, q_j, K) * (n, q_j, 1) -> (n, q_j, K)
    contribution = param_rv[group_idx_data] * predictor_data
    if dims_expr:
        # (n, q_j) -> (n, )
        # (n, q_j, K) -> (n, K)
        contribution = contribution.sum(axis=1)

    # NOTE: This returns something already done, the others return multiple things
    return contribution


def build_distribution(prior, label, dims_factor, dims_expr, dims_output, noncentered, model):
    kwargs = {}
    for name, value in prior.args.items():
        if isinstance(value, Prior):
            hyperparam_name = name
            hyperparam_label = f"{label}_{hyperparam_name}"
            kwargs[name] = build_distribution(
                prior=value,
                label=hyperparam_label,
                dims_factor=tuple(),
                dims_expr=dims_expr,
                dims_output=dims_output,
                noncentered=noncentered,
                model=model,
            )
        else:
            kwargs[name] = value

    # From lowest to fastest changing
    dims = dims_factor + dims_expr + dims_output
    if noncentered and any(isinstance(v, pt.TensorVariable) for v in kwargs.values()):
        # non-centered is only relevant when distribution arguments are random variables.
        if prior.name == "Normal" and isinstance(kwargs.get("sigma", None), pt.TensorVariable):
            sigma = kwargs["sigma"]
            offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims, model=model)
            return pm.Deterministic(label, offset * sigma, dims=dims, model=model)

        raise NotImplementedError(
            "The non-centered parametrization is only supported for Normal priors"
        )

    dist = get_distribution_from_prior(prior)
    return dist(label, **kwargs, dims=dims, model=model)


# NOTE: aliases should be managed by Bambi and how its terms return names, not by anything in PyMC.
