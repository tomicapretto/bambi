import operator

import formulae as fm
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.model.transform.basic import prune_vars_detached_from_observed
from pymc.pytensorf import toposort_replace
from pytensor.graph.traversal import ancestors

from bambi.backend.pymc.terms import (
    build_common_term,
    build_group_specific_term_dot,
    build_group_specific_term_idx,
    build_hsgp_term,
    build_intercept_term,
)
from bambi.backend.pymc.utils import INVERSE_LINKS
from bambi.backend.pymc.transform import transforms_registry
from bambi.backend.pymc.coords import coords_from_common, coords_from_group_specific
from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.config import config as bmb_config
from bambi.families import Family
from bambi.families.types import ParamSpec


_ENSURE_NDIM_MAPPING = {
    0: pt.atleast_1d,
    1: pt.atleast_2d,
}

_GROUP_SPECIFIC_TAG = "group_specific_contribution"


def new_group_selector_name(factor_name: str) -> str:
    """Return the internal posterior variable used to select a new group per draw."""
    return f"__new_group_{factor_name}_selector"


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


def _build_group_specific_idx(terms, param_spec: ParamSpec, model: pm.Model):
    contribution = 0
    for term in terms.values():
        contribution += build_group_specific_term_idx(term, param_spec, model)
    return contribution


def build_omitted_group_offsets(parameter):
    """Build recipes for reconstructing omitted non-centered group offsets."""
    offsets = {}
    for term in parameter.group_specific_terms.values():
        if term.noncentered:
            offsets[f"{term.label}_offset"] = (
                operator.truediv,
                term.label,
                f"{term.label}_sigma",
            )
    return offsets


def remove_group_specific_contributions(parameters, model: pm.Model) -> pm.Model:
    """Return a model without tagged group-specific predictor contributions.

    Removes the complete subgraph for the group-specific contribution
    while leaving coefficient variables and the fitted model untouched.
    """
    fgraph, memo = fgraph_from_model(model)
    replacements = []

    for parameter in parameters:
        if not parameter.group_specific_terms:
            continue

        label = parameter.label
        contribution = next(
            (
                variable
                for variable in ancestors([model[label]])
                if getattr(variable.tag, _GROUP_SPECIFIC_TAG, None) == label
            ),
            None,
        )
        if contribution is None:
            raise ValueError(f"Could not find group-specific contribution for parameter '{label}'.")

        term = next(iter(parameter.group_specific_terms.values()))
        coords_expr, coords_factor = coords_from_group_specific(term)
        term_dims = model.named_vars_to_dims[term.label]
        output_dims_start = len(coords_factor) + len(coords_expr)
        dims = ("__obs__", *term_dims[output_dims_start:])

        contribution = memo[contribution]
        shape = tuple(memo.get(model.dim_lengths[dim], model.dim_lengths[dim]) for dim in dims)
        replacements.append((contribution, pt.zeros(shape, dtype=contribution.dtype)))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return prune_vars_detached_from_observed(model)


def add_new_group_specific_contributions(parameters, model: pm.Model, new_groups) -> pm.Model:
    """Allow a prediction clone to look up one sampled coefficient per unseen factor.

    ``new_groups`` maps grouping-factor names to their number of fitted levels. Data for an
    unseen level uses that number as a sentinel index. This function adds a one-element
    extension to the corresponding coefficient array, whose value is selected from a fitted
    level by a scalar variable supplied in the posterior trace for each draw.
    """
    selectors = {}
    with model:
        for factor_name in new_groups:
            selectors[factor_name] = pm.Flat(new_group_selector_name(factor_name))

    fgraph, memo = fgraph_from_model(model)
    replacements = []

    if bmb_config["SPARSE_DOT"]:
        for parameter in parameters:
            if not parameter.group_specific_terms:
                continue

            contribution = next(
                (
                    variable
                    for variable in ancestors([model[parameter.label]])
                    if getattr(variable.tag, _GROUP_SPECIFIC_TAG, None) == parameter.label
                ),
                None,
            )
            dot_output = next(
                (
                    variable
                    for variable in ancestors([contribution])
                    if variable.owner is not None
                    and type(variable.owner.op).__name__ == "StructuredDot"
                ),
                None,
            )
            if dot_output is None:
                raise ValueError(
                    "Could not find the sparse group-specific contribution for "
                    f"parameter '{parameter.label}'."
                )

            coefs = dot_output.owner.inputs[1]
            # Univariate terms are expanded to satisfy ``structured_dot``.
            while coefs.owner is not None and type(coefs.owner.op).__name__ in {
                "ExpandDims",
                "DimShuffle",
            }:
                coefs = coefs.owner.inputs[0]

            if coefs.owner is not None and type(coefs.owner.op).__name__ == "Join":
                coef_blocks = coefs.owner.inputs[1:]
            else:
                coef_blocks = (coefs,)

            terms = tuple(parameter.group_specific_terms.values())
            if len(coef_blocks) != len(terms):
                raise ValueError(
                    "Could not align sparse group-specific coefficient blocks with their terms."
                )

            extended_blocks = []
            for term, coef_block in zip(terms, coef_blocks):
                coef_block = memo[coef_block]
                if term.factor_name not in new_groups:
                    extended_blocks.append(coef_block)
                    continue

                n_levels = new_groups[term.factor_name]
                block_size = term.data.shape[1] // n_levels
                shape = (n_levels, block_size, *coef_block.shape[1:])
                coefficients_by_group = coef_block.reshape(shape)
                selector = pt.cast(memo[selectors[term.factor_name]], "int64")
                new_group_coef = coefficients_by_group[selector]
                extended_blocks.append(pt.concatenate([coef_block, new_group_coef], axis=0))

            extended_coefs = pt.concatenate(extended_blocks, axis=0)
            dot_output = memo[dot_output]
            data, dot_coefs = dot_output.owner.inputs
            if extended_coefs.ndim < dot_coefs.ndim:
                extended_coefs = pt.shape_padright(
                    extended_coefs, dot_coefs.ndim - extended_coefs.ndim
                )
            replacements.append((dot_output, ps.structured_dot(data, extended_coefs)))

        if not replacements:
            raise ValueError("Could not find group-specific contributions for unseen groups.")

        toposort_replace(fgraph, replacements, reverse=True)
        model = model_from_fgraph(fgraph, mutate_fgraph=True)
        return prune_vars_detached_from_observed(model)

    for parameter in parameters:
        if not parameter.group_specific_terms:
            continue

        for variable in ancestors([model[parameter.label]]):
            factor_name = getattr(variable.tag, "group_specific_factor", None)
            if factor_name not in new_groups:
                continue

            selector = pt.cast(memo[selectors[factor_name]], "int64")
            if variable.owner is None or len(variable.owner.inputs) != 2:
                raise ValueError(
                    "Could not replace the group-specific coefficient lookup for "
                    f"factor '{factor_name}'."
                )

            selected_param = memo[variable]
            coefficients, group_idx = selected_param.owner.inputs
            new_group_coef = coefficients[selector]
            extended_coefficients = pt.concatenate(
                [coefficients, pt.shape_padleft(new_group_coef)], axis=0
            )
            replacements.append((selected_param, extended_coefficients[group_idx]))

    if not replacements:
        raise ValueError("Could not find group-specific contributions for unseen groups.")

    toposort_replace(fgraph, replacements, reverse=True)
    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return prune_vars_detached_from_observed(model)


def build_conditional_parameter(parameter, family: Family, model: pm.Model):
    value = 0
    param_spec = family.get_param_spec(parameter.name)
    link = family.link[parameter.name]
    inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
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
        group_specific_contribution = _build_group_specific(
            terms=parameter.group_specific_terms, param_spec=param_spec, model=model
        )
        setattr(group_specific_contribution.tag, _GROUP_SPECIFIC_TAG, parameter.label)
        value += group_specific_contribution

    for term in parameter.hsgp_terms.values():
        value += build_hsgp_term(term, param_spec, model)

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


def build_new_conditional_parameter_data(parameter, data, model: pm.Model, new_groups):
    """Build new-observation data and record unseen grouping factors.

    Formulae represents unseen categories as all-zero dummy rows in ``silent`` mode. Those rows
    receive a sentinel group index; the prediction model later replaces the sentinel lookup with
    coefficients sampled from the posterior.
    """
    data_dict = {}

    for term in parameter.common_terms.values():
        coords = coords_from_common(term)
        term_data_name = predictor_data_name(term.label, ("__obs__", *coords), model)
        term_data_dims = model.named_vars_to_dims[term_data_name][1:]  # drop __obs__
        term_data = shape_common_data(
            data=term.term.eval_new_data(data),
            coords={dim: model.coords[dim] for dim in term_data_dims},
        )
        data_dict.update({term_data_name: term_data})

    for term in parameter.group_specific_terms.values():
        original_unseen_config = fm.config["EVAL_UNSEEN_CATEGORIES"]
        fm.config["EVAL_UNSEEN_CATEGORIES"] = "silent"
        try:
            factor_data = term.factor.eval_new_data(data)
            term_data = term.term.eval_new_data(data) if bmb_config["SPARSE_DOT"] else None
        finally:
            fm.config["EVAL_UNSEEN_CATEGORIES"] = original_unseen_config

        unseen_rows = ~factor_data.any(axis=1)
        n_levels = int(term.group_index.max()) + 1
        if unseen_rows.any():
            known_n_levels = new_groups.setdefault(term.factor_name, n_levels)
            if known_n_levels != n_levels:
                raise ValueError(
                    f"Inconsistent group-level counts for factor '{term.factor_name}'."
                )

        if bmb_config["SPARSE_DOT"]:
            term_data_name = f"{term.label}_data"
            data_dict[term_data_name] = term_data
        else:
            term_idx_name = f"{term.factor_name}__idx"
            term_idx_data = term.invert_dummies(factor_data)
            if unseen_rows.any():
                term_idx_data = term_idx_data.copy()
                term_idx_data[unseen_rows] = n_levels
            data_dict[term_idx_name] = term_idx_data

            if not term.is_intercept:
                coords_expr, _ = coords_from_group_specific(term)
                term_value_name = predictor_data_name(
                    term.expr_name, ("__obs__", *coords_expr), model
                )
                term_value_dims = model.named_vars_to_dims[term_value_name][1:]  # drop __obs__
                term_value_data = shape_common_data(
                    data=term.expr.eval_new_data(data),
                    coords={dim: model.coords[dim] for dim in term_value_dims},
                )
                data_dict[term_value_name] = term_value_data

    for term in parameter.hsgp_terms.values():
        term_data = term.term.eval_new_data(data)
        if term.by_levels is not None:
            by_data = term_data[:, -1].astype(int)
            term_data = term_data[:, :-1]
            data_dict[f"{term.label}_by_idx"] = by_data
        data_dict[f"{term.label}_data"] = term_data

    return data_dict
