import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.pymc.coords import coords_from_group_specific
from bambi.backend.pymc.data import shape_common_data
from bambi.backend.pymc.terms.common import shape_prior_arg
from bambi.backend.pymc.types import Dims
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.priors.prior import Prior
from bambi.families.types import ParamSpec


# NOTE: Can we assume data_name is unique?
#       How do we manage the case where we have `x` and `sigma_x`?
#       That woud cause `data_name` to be in conflict.
def build_group_specific_term_dot(
    term, param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, pt.Variable]:
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
    model.add_coords({data_dims[1]: range(data.shape[1])})
    pm.Data(data_name, data, dims=data_dims, model=model)

    # Register parameter
    dims_output = tuple()
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims_output = tuple(model.__bambi_attrs__["response_coordsd"])
        elif param_spec.coefs_dim == "response_reduced":
            dims_output = tuple(model.__bambi_attrs__["response_coords_reduced"])

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


def build_group_specific_term_idx(term, param_spec: ParamSpec, model: pm.Model) -> pt.Variable:
    data_value_name = f"{term.label}_data"
    data_idx_name = f"{term.label}_idx"
    param_name = term.label

    coords_expr, coords_factor = coords_from_group_specific(term)
    coords = coords_factor | coords_expr
    dims_expr = tuple(coords_expr)
    dims_factor = tuple(coords_factor)

    # Register coords
    # Data is not checked as no coords are registered for it
    if param_name not in model:
        model.add_coords(coords)

    # Register data, predictor
    predictor_dims = ("__obs__",) + dims_expr
    predictor = shape_common_data(term.predictor, coords_expr)
    predictor_data = pm.Data(data_value_name, predictor, dims=predictor_dims, model=model)

    # Register data, group index (which index of parameter to select from)
    group_idx_data = pm.Data(data_idx_name, term.group_index, dims=("__obs__",), model=model)

    # Register parameter
    dims_output = tuple()
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims_output = tuple(model.__bambi_attrs__["response_coordsd"])
        elif param_spec.coefs_dim == "response_reduced":
            dims_output = tuple(model.__bambi_attrs__["response_coords_reduced"])

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
        predictor_data = predictor_data[..., np.newaxis]

    # (n, ) * (n, )             -> (n, )
    # (n, q_j) * (n, q_j)       -> (n, q_j)
    # (n, K) * (n, 1)           -> (n, K)
    # (n, q_j, K) * (n, q_j, 1) -> (n, q_j, K)
    contribution = param_rv[group_idx_data] * predictor_data
    if dims_expr:
        axes = tuple(range(1, len(dims_expr) + 1))
        contribution = contribution.sum(axis=axes)

    # NOTE: This returns something already in final state, the others return multiple things
    return contribution


def build_distribution(
    prior: Prior,
    label: str,
    dims_factor: Dims,
    dims_expr: Dims,
    dims_output: Dims,
    noncentered: bool,
    model: pm.Model,
) -> pt.Variable:
    kwargs = {}
    # From slowest to fastest changing
    dims = dims_factor + dims_expr + dims_output
    shape = tuple(len(model.coords[dim]) for dim in dims)

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
            kwargs[name] = shape_prior_arg(value, shape)

    if noncentered and any(isinstance(v, pt.TensorVariable) for v in kwargs.values()):
        # non-centered is only relevant when distribution arguments are random variables.
        if prior.name == "Normal" and isinstance(kwargs.get("sigma", None), pt.TensorVariable):
            sigma = kwargs["sigma"]
            with model:
                offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims)
                rv = pm.Deterministic(label, offset * sigma, dims=dims)
            return rv

        raise NotImplementedError(
            "The non-centered parametrization is only supported for Normal priors"
        )

    dist = get_distribution_from_prior(prior)

    with model:
        rv = dist(label, **kwargs, dims=dims)

    return rv
