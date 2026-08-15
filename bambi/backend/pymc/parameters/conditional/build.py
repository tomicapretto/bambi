import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

from bambi.backend.pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_group_specific_term_idx,
    build_hsgp_term,
    build_intercept_term,
)
from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.config import config as bmb_config
from bambi.families import Family
from bambi.families.types import ParamSpec
from bambi.terms import CommonTerm

from .state import (
    CommonTermInfo,
    ConditionalParameterInfo,
    GroupSpecificGraphState,
    GroupSpecificParameterGraph,
    GroupSpecificTermGraph,
    GroupSpecificTermInfo,
)


def build_conditional_parameter(
    parameter_info: ConditionalParameterInfo,
    family: Family,
    model: pm.Model,
    graph_state: GroupSpecificGraphState,
) -> pt.Variable:
    parameter = parameter_info.parameter
    value = 0
    param_spec = family.get_param_spec(parameter.name)
    link = family.link[parameter.name]
    inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
    center_predictors = parameter.intercept_term and parameter.center_predictors

    if parameter_info.common_terms or parameter.intercept_term:
        value += _build_common_and_intercept(
            common_terms=parameter_info.common_terms,
            intercept_term=parameter.intercept_term,
            center=center_predictors,
            param_spec=param_spec,
            model=model,
        )

    if parameter_info.group_specific_terms:
        group_specific_contribution = _build_group_specific(
            parameter_info=parameter_info,
            param_spec=param_spec,
            model=model,
            graph_state=graph_state,
        )
        value += group_specific_contribution

    for term_info in parameter_info.hsgp_terms:
        value += build_hsgp_term(term_info, param_spec, model)

    # TODO: Make sure parameters are built in the appropriate order
    transform_predictor = transforms_registry.get_predictor_transform(family, parameter.name)
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
        value = pt.broadcast_to(value, tuple(model.dim_lengths[dim] for dim in dims))
    return pm.Deterministic(parameter.label, value, dims=dims, model=model)


_ENSURE_NDIM_MAPPING = {
    0: pt.atleast_1d,
    1: pt.atleast_2d,
}


def _ensure_2d(x: pt.Variable) -> pt.Variable:
    # Concatenation requires data arrays to be all 2d
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _build_common_and_intercept(
    common_terms: tuple[CommonTermInfo, ...],
    intercept_term: CommonTerm | None,
    center: bool,
    param_spec: ParamSpec,
    model: pm.Model,
) -> pt.Variable:
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

        for term_info in common_terms:
            data, param = build_common_term(term_info, param_spec, model)
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


def _build_group_specific(
    parameter_info: ConditionalParameterInfo,
    param_spec: ParamSpec,
    model: pm.Model,
    graph_state: GroupSpecificGraphState,
) -> pt.Variable:
    terms = parameter_info.group_specific_terms
    contribution_dims = ("__obs__",)

    if param_spec.coefs_dim == "response":
        contribution_dims += tuple(model.__bambi_attrs__["response_coords"])
    elif param_spec.coefs_dim == "response_reduced":
        contribution_dims += tuple(model.__bambi_attrs__["response_coords_reduced"])

    if bmb_config["SPARSE_DOT"]:
        contribution = _build_group_specific_dot(terms, param_spec, model)
        graph_state.parameters[parameter_info.label] = GroupSpecificParameterGraph(
            contribution=contribution,
            contribution_dims=contribution_dims,
        )
        return contribution

    contribution, term_graphs = _build_group_specific_idx(terms, param_spec, model)
    graph_state.parameters[parameter_info.label] = GroupSpecificParameterGraph(
        contribution=contribution,
        contribution_dims=contribution_dims,
        terms=term_graphs,
    )
    return contribution


def _build_group_specific_dot(
    terms: tuple[GroupSpecificTermInfo, ...], param_spec: ParamSpec, model: pm.Model
) -> pt.Variable:
    data_blocks = []
    param_blocks = []
    for term_info in terms:
        term = term_info.term
        data, param = build_group_specific_term_dot(term, param_spec, model)
        data_blocks.append(data)
        param_blocks.append(param)

    # Design matrix Z: shape (n, q)
    data = ps.hstack(data_blocks, format="csr")

    # Coefficients array: shape (q, ) or (q, K)
    coefs = pt.concatenate(param_blocks, axis=0)

    is_univariate = coefs.ndim == 1
    if is_univariate:
        # PyTensor expects 2D
        coefs = coefs[:, np.newaxis]

    # (n, ) or (n, K)
    dot_output = ps.structured_dot(data, coefs)
    if is_univariate:
        return dot_output.squeeze()

    return dot_output


def _build_group_specific_idx(
    terms: tuple[GroupSpecificTermInfo, ...], param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, dict[str, GroupSpecificTermGraph]]:
    contribution = 0
    term_graphs = {}
    for term_info in terms:
        term = term_info.term
        lookup, term_contribution = build_group_specific_term_idx(term, param_spec, model)
        term_graphs[term.label] = GroupSpecificTermGraph(lookup=lookup)
        contribution += term_contribution
    return contribution, term_graphs
