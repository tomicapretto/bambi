import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pymc.model.transform.basic import prune_vars_detached_from_observed
from pymc.pytensorf import replace_vars_in_graphs, toposort_replace

from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.terms.common import shape_prior_arg
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.config import config as bmb_config
from bambi.priors.prior import Prior

from .state import (
    ConditionalParameterInfo,
    GroupSpecificFactorPlan,
    GroupSpecificGraphState,
    GroupSpecificTermInfo,
)


def remove_group_specific_contributions(
    model: pm.Model, graph_state: GroupSpecificGraphState
) -> pm.Model:
    """Discard group-specific branches from a model clone and prune the detached variables."""
    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    replacements = []

    for parameter_graph in graph_state.parameters.values():
        contribution = memo[parameter_graph.contribution]
        contribution_dims = parameter_graph.contribution_dims
        shape = tuple(memo[model.dim_lengths[dim]] for dim in contribution_dims)
        replacements.append((contribution, pt.zeros(shape, dtype=contribution.dtype)))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return prune_vars_detached_from_observed(model)


def add_new_group_specific_contributions(
    model: pm.Model,
    plans: list[GroupSpecificFactorPlan],
    graph_state: GroupSpecificGraphState,
) -> pm.Model:
    """Replace dense group-specific lookups for out-of-sample groups."""
    if bmb_config["SPARSE_DOT"]:
        raise NotImplementedError(
            "Sampling out-of-sample group-specific effects with SPARSE_DOT=True is not yet "
            "implemented."
        )

    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    replacements = []

    for plan in plans:
        group_idx = model[f"{plan.factor_name}__idx"]
        effective_idx = group_idx
        # `groups_index` uses -1 for missing levels
        unknown_mask = plan.groups_index == -1

        if unknown_mask.any():
            # Share donors across terms to preserve their posterior association.
            p = np.full(plan.groups_n, 1 / plan.groups_n)
            donor_idx = pm.Categorical.dist(p=p, shape=plan.groups_index.shape[0])
            effective_idx = pt.where(unknown_mask, donor_idx, group_idx)

        for term_info in plan.terms:
            term = term_info.term
            coefficients = model[term.label]
            if plan.factor_ndim > 1:
                # `effective_idx` is flat, including for missing-level donors.
                tail_shape = tuple(
                    coefficients.shape[i] for i in range(plan.factor_ndim, coefficients.ndim)
                )
                coefficients = coefficients.reshape((-1, *tail_shape))

            if plan.groups_new:
                new_coefficients = _create_new_group_coefficients(
                    term_info, len(plan.groups_new), plan.factor_ndim, model
                )
                coefficients = pt.concatenate([coefficients, new_coefficients], axis=0)

            replacement = replace_vars_in_graphs([coefficients[effective_idx]], memo)[0]
            lookup = memo[graph_state.parameters[plan.parameter_label].terms[term.label].lookup]
            replacements.append((lookup, replacement))

    if replacements:
        toposort_replace(fgraph, replacements, reverse=True)

    return model_from_fgraph(fgraph, mutate_fgraph=True)


def build_new_conditional_parameter_data(
    parameter_info: ConditionalParameterInfo, data: pd.DataFrame, model: pm.Model
):
    """Build new-observation data and a prediction plan per grouping factor."""
    parameter = parameter_info.parameter
    data_dict = {}
    factor_plans = []

    for term_info in parameter_info.common_terms:
        term = term_info.term
        term_data_name = predictor_data_name(term.label, term_info.data_dims, model)
        term_data_dims = model.named_vars_to_dims[term_data_name][1:]  # drop __obs__
        term_data = shape_common_data(
            data=term.term.eval_new_data(data),
            coords={dim: model.coords[dim] for dim in term_data_dims},
        )
        data_dict.update({term_data_name: term_data})

    for factor_info in parameter_info.group_specific_factors:
        representative = factor_info.terms[0].term
        group_index, new_groups = representative.term.eval_new_data_group_index(data)
        factor_plans.append(
            GroupSpecificFactorPlan(
                parameter_label=parameter.label,
                factor_name=factor_info.factor_name,
                factor_ndim=factor_info.factor_ndim,
                terms=factor_info.terms,
                groups_index=group_index,
                groups_new=new_groups,
                groups_n=factor_info.groups_n,
            )
        )

        # NOTE: Out-of-sample SPARSE_DOT prediction is not implemented.
        if bmb_config["SPARSE_DOT"]:
            # Out-of-sample groups are rejected before this data are installed, so there is no
            # need to ask Formulae to evaluate their group-specific design matrices.
            if (group_index == -1).any() or new_groups:
                continue
            for term_info in factor_info.terms:
                term = term_info.term
                data_dict[f"{term.label}_data"] = term.term.eval_new_data(data)
            continue

        term_idx_name = f"{factor_info.factor_name}__idx"
        data_dict[term_idx_name] = group_index

        for term_info in factor_info.terms:
            term = term_info.term
            if term.is_intercept:
                continue
            term_value_name = predictor_data_name(
                term.expr_name, ("__obs__", *term_info.expression_coords), model
            )
            term_value_dims = model.named_vars_to_dims[term_value_name][1:]  # drop __obs__
            term_value_data = shape_common_data(
                data=term.expr.eval_new_data(data),
                coords={dim: model.coords[dim] for dim in term_value_dims},
            )
            data_dict[term_value_name] = term_value_data

    for term_info in parameter_info.hsgp_terms:
        term = term_info.term
        term_data = term.term.eval_new_data(data)
        if term.by_levels is not None:
            by_data = term_data[:, -1].astype(int)
            term_data = term_data[:, :-1]
            data_dict[f"{term.label}_by_idx"] = by_data
        data_dict[f"{term.label}_data"] = term_data

    return data_dict, factor_plans


def _create_new_group_coefficients(
    term_info: GroupSpecificTermInfo,
    n_new_groups: int,
    factor_ndim: int,
    model: pm.Model,
) -> pt.Variable:
    """Create unregistered population draws for newly identified group levels."""
    term = term_info.term
    term_dims = model.named_vars_to_dims[term.label]
    # Keep expression and response axes after replacing factor axes with new groups.
    tail_dims = term_dims[factor_ndim:]
    tail_shape = tuple(len(model.coords[dim]) for dim in tail_dims)
    kwargs = {}

    for name, value in term.prior.args.items():
        if isinstance(value, Prior):
            # Reuse hyperprior RVs from the fitted model.
            kwargs[name] = model[f"{term.label}_{name}"]
        else:
            # Match fixed prior arguments to the retained axes.
            kwargs[name] = shape_prior_arg(value, tail_shape)

    # Create unregistered draws, one per new group.
    distribution = get_distribution_from_prior(term.prior)
    return distribution.dist(**kwargs, size=(n_new_groups, *tail_shape))
