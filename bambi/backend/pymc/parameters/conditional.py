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
    if model.__bambi_attrs__["parameter_ndim"] == 1:
        return pt.atleast_1d
    return pt.atleast_2d


def _ensure_2d(x):
    # Concatenation requires data arrays to be all 2d
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _build_intercept(term, data_mean, common_params, model):
    ensure_ndim = _get_ensure_ndim(model)
    return ensure_ndim(build_intercept_term(term, data_mean, common_params, model))


def _build_common(terms, center, model):
    data_list = []
    param_list = []
    ensure_ndim = _get_ensure_ndim(model)

    for term in terms.values():
        data, param = build_common_term(term, model)
        data_list.append(_ensure_2d(data))
        param_list.append(ensure_ndim(param))

    params = pt.concatenate(param_list, axis=0)  # (p, ) or (p, K)
    data = pt.concatenate(data_list, axis=1)  # (n, p)

    if center:
        data_mean = data.mean(0)
        data = data - data_mean
    else:
        data_mean = None

    # (n, ) or (n, K)
    contribution = pt.dot(data, params)
    return contribution, data_mean, params


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
    common_data_mean = None
    common_params = None
    inverse_link = INVERSE_LINKS.get(family.link[parameter.name].name, lambda x: x)
    center_predictors = parameter.intercept_term and parameter.center_predictors

    # Common terms are built before the intercept so we can uncenter the intercept later.
    if parameter.common_terms:
        contribution, common_data_mean, common_params = _build_common(
            parameter.common_terms, center_predictors, model
        )
        value += contribution

    if parameter.intercept_term:
        value += _build_intercept(parameter.intercept_term, common_data_mean, common_params, model)

    if parameter.group_specific_terms:
        value += _build_group_specific(parameter.group_specific_terms, model)

    # TODO: Make sure parameters are built in the appropriate order
    transform_predictor = transforms_registry.get_transform_predictor(family, parameter.name)
    if transform_predictor:
        parameters = {
            name: model[name] for name in family.likelihood.params if name != parameter.name
        }
        value = transform_predictor(value, parameters, inverse_link)
    else:
        value = inverse_link(value)

    coords = (
        model.__bambi_attrs__["response_coords_data"] | model.__bambi_attrs__["response_coords"]
    )
    dims = tuple(coords)
    only_intercept = (
        parameter.intercept_term
        and not parameter.common_terms
        and not parameter.group_specific_terms
        and not parameter.offset_terms
        and not parameter.hsgp_terms
    )
    if value.ndim < len(dims) or only_intercept:
        value = pt.broadcast_to(value, tuple(len(coord) for coord in coords.values()))
    return pm.Deterministic(parameter.label, value, dims=dims, model=model)
