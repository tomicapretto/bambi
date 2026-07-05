import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
import scipy as sp

from bambi.backend.pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_group_specific_term_idx,
    build_intercept_term,
)
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.backend.pymc.transform import transforms_registry
from bambi.config import config as bmb_config
from bambi.families import Family
from bambi.families.types import ParamSpec


_ENSURE_NDIM_MAPPING = {
    0: pt.atleast_1d,
    1: pt.atleast_2d,
}


def _ensure_2d(x):
    # Concatenation requires data arrays to be all 2d
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _build_common_and_intercept(
    common_terms, intercept_term, center: bool, param_spec: ParamSpec, model: pm.Model
):
    # Build common terms, then build intercept
    ndim = 0 if param_spec.coefs_dim is None else 1
    ensure_ndim = _ENSURE_NDIM_MAPPING[ndim]
    data_mean = None
    params = None
    intercept_contribution = 0
    common_contribution = 0

    if common_terms:
        data_list = []
        param_list = []

        for term in common_terms.values():
            data, param = build_common_term(term, param_spec, model)
            data_list.append(_ensure_2d(data))
            param_list.append(ensure_ndim(param))

        params = pt.concatenate(param_list, axis=0)  # (p, ) or (p, K)
        data = pt.concatenate(data_list, axis=1)  # (n, p)

        if center:
            data_mean = data.mean(0)
            data = data - data_mean

        # (n, ) or (n, K)
        common_contribution = pt.dot(data, params)

    if intercept_term:
        intercept_contribution = ensure_ndim(
            build_intercept_term(intercept_term, data_mean, params, param_spec, model)
        )

    return intercept_contribution + common_contribution


def _build_group_specific(terms, param_spec: ParamSpec, model: pm.Model):
    if bmb_config["SPARSE_DOT"]:
        return _build_group_specific_dot(terms=terms, param_spec=param_spec, model=model)
    return _build_group_specific_idx(terms=terms, param_spec=param_spec, model=model)


def _build_group_specific_dot(terms, param_spec: ParamSpec, model: pm.Model):
    data_blocks = []
    param_blocks = []
    for term in terms.values():
        data, param = build_group_specific_term_dot(term, param_spec, model)
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


def _build_group_specific_idx(terms, param_spec: ParamSpec, model: pm.Model):
    contribution = 0
    for term in terms.values():
        contribution += build_group_specific_term_idx(term, param_spec, model)
    return contribution


def build_conditional_parameter(parameter, family: Family, model: pm.Model):
    # NOTE: `param_spec` does not work with families that have None in PARAMETERS.
    value = 0
    param_spec = family.PARAMETERS[parameter.name]
    inverse_link = INVERSE_LINKS.get(family.link[parameter.name].name, lambda x: x)
    center_predictors = parameter.intercept_term and parameter.center_predictors

    if parameter.common_terms or parameter.intercept_term:
        value += _build_common_and_intercept(
            common_terms=parameter.common_terms,
            intercept_term=parameter.intercept_term,
            center=center_predictors,
            param_spec=param_spec,
            model=model,
        )

    if parameter.group_specific_terms:
        value += _build_group_specific(
            terms=parameter.group_specific_terms, param_spec=param_spec, model=model
        )

    # TODO: Make sure parameters are built in the appropriate order
    transform_predictor = transforms_registry.get_transform_predictor(family, parameter.name)
    if transform_predictor:
        parameters = {
            name: model[name] for name in family.likelihood.params if name != parameter.name
        }
        value = transform_predictor(value, parameters, inverse_link)
    else:
        value = inverse_link(value)

    coords = model.__bambi_attrs__["response_coords_data"]
    if param_spec.ndim > 0:
        coords = coords | model.__bambi_attrs__["response_coords"]

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
