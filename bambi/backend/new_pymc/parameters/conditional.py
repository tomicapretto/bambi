import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
import scipy as sp

from bambi.backend.new_pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_intercept_term,
    build_group_specific_term_idx,
)
from bambi.config import config as bmb_config

INVLINKS = {}


def _get_ensure_ndim(model):
    if model.__bambi_attrs__["output_ndim"] == 1:
        return pt.atleast_1d
    return pt.atleast_2d


def _build_intercept(term, model):
    ensure_ndim = _get_ensure_ndim(model)
    return ensure_ndim(build_intercept_term(term, model))


def _build_common(terms, center, model):
    data_list = []
    param_list = []
    ensure_ndim = _get_ensure_ndim(model)

    for term in terms.values():
        data, param = build_common_term(term, model)
        data_list.append(data)
        param_list.append(ensure_ndim(param))

    params = pt.concatenate(param_list, axis=0)  # (p, ) or (p, K)
    data = pt.concatenate(data_list, axis=1)  # (n, p)

    if center:
        data = data - data.mean(0)

    # (n, ) or (n, K)
    return pt.dot(data, params)


def _build_group_specific(terms, model):
    if bmb_config["SPARSE_DOT"]:
        return _build_group_specific_dot(terms, model)
    return _build_group_specific_idx(terms, model)


def _build_group_specific_dot(terms, model):
    data_blocks = []
    param_blocks = []
    for term in terms.values():
        data, param = build_group_specific_term_dot(term, model)
        data_blocks.append(data)
        param_blocks.append(param)

    # Design matrix Z: shape (n, q)
    data = sp.sparse.hstack(data_blocks, format="csr")

    # Coefficients array: shape (q, ) or (q, K)
    coefs = pt.concatenate(param_blocks, axis=0)

    if coefs.ndim == 1:
        # PyTensor expects 2D
        coefs = coefs[:, np.newaxis]

    # (n, ) or (n, K)
    # FIXME: Do we always need to squeeze?
    return ps.structured_dot(data, coefs).squeeze()


def _build_group_specific_idx(terms, model):
    contribution = 0
    for term in terms.values():
        contribution += build_group_specific_term_idx(term, model)
    return contribution


def build_conditional_parameter(parameter, family, model):
    value = 0
    if parameter.intercept_term:
        value += _build_intercept(parameter.intercept_term, model)

    if parameter.common_terms:
        # TODO: parameter.center_predictors should query the bambi model
        center = parameter.intercept_term and parameter.center_predictors
        value += _build_common(parameter.common_terms, center, model)

    if parameter.group_specific_terms:
        value += _build_group_specific(parameter.group_specific_terms, model)

    # TODO: I move on as if parameters were already in place. This is necessarily not true
    # We can specify dependencies between parameters in the model family, and build them
    # in the appropriate order.
    if hasattr(family, f"transform_{parameter.name}"):
        transform_parameter = getattr(family, f"transform_{parameter.name}")
        parameters_mapping = {
            name: model[name] for name in family.likelihood.parameters if name != parameter.name
        }
        value = transform_parameter(value, parameters_mapping)

    linkinv = get_linkinv(family.link[parameter.name], INVLINKS)

    rv = pm.Deterministic(
        parameter.label,
        linkinv(value),
        dims=tuple(model.__bambi_attrs__["output_coords"]),
        model=model,
    )

    return rv


def get_linkinv(link, invlinks):
    """Get the inverse of the link function as needed by PyMC

    Parameters
    ----------
    link : bmb.Link
        A link function object. It may contain the linkinv function that the backend uses.
    invlinks : dict
        Keys are names of link functions. Values are the built-in link functions.

    Returns
    -------
        callable
        The link function.
    """
    # If the name is in the backend, get it from there
    if link.name in invlinks:
        invlink = invlinks[link.name]
    # If not, use whatever is in `linkinv_backend`
    else:
        invlink = link.linkinv_backend
    return invlink
