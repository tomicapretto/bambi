import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
import scipy as sp

from bambi.backend.pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_intercept_term,
    build_group_specific_term_idx,
)
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.config import config as bmb_config
from bambi.backend.pymc.transform import transforms_registry


def _get_ensure_ndim(model):
    if model.__bambi_attrs__["response_ndim"] == 1:
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

    # TODO: Register as a deterministic
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
    dot_output = ps.structured_dot(data, coefs)
    if coefs.ndim == 1:
        return dot_output.squeeze()

    return dot_output


def _build_group_specific_idx(terms, model):
    contribution = 0
    for term in terms.values():
        contribution += build_group_specific_term_idx(term, model)
    return contribution


def build_conditional_parameter(parameter, family, model):
    value = 0
    inverse_link = INVERSE_LINKS.get(family.link[parameter.name].name, lambda x: x)

    if parameter.intercept_term:
        value += _build_intercept(parameter.intercept_term, model)

    if parameter.common_terms:
        # TODO: parameter.center_predictors should query the bambi model
        center = parameter.intercept_term and parameter.center_predictors
        value += _build_common(parameter.common_terms, center, model)

    if parameter.group_specific_terms:
        value += _build_group_specific(parameter.group_specific_terms, model)

    # TODO: Make sure parameters are built in the appropriate order
    transform_parameter = transforms_registry.get_transform_parameters(family)
    if transform_parameter:
        parameters = {
            name: model[name] for name in family.likelihood.parameters if name != parameter.name
        }
        value = transform_parameter(value, parameters, inverse_link)
    else:
        value = inverse_link(value)

    dims = tuple(
        model.__bambi_attrs__["response_coords_data"] | model.__bambi_attrs__["response_coords"]
    )
    return pm.Deterministic(parameter.label, value, dims=dims, model=model)
