import pymc as pm
import pytensor.tensor as pt

from bambi.priors import Prior
from bambi.backend.new_pymc.coords import coords_from_group_specific
from bambi.backend.new_pymc.utils import get_distribution_from_prior


def build_group_specific_term(term, model):
    # FIXME WIP
    coords_expr, coords_factor = coords_from_group_specific(term)
    dims_expr = tuple(coords_expr) or None
    dims_factor = tuple(coords_factor) or None

    # Register data
    data = ...

    # Register parameter
    parameter = _build_distribution(
        prior=term.prior,
        label=term.name,
        dims_expr=dims_expr,
        dims_facotr=dims_factor,
        noncentered=term.noncentered,
        model=model,
    )

    return None  # FIXME


def _build_distribution(prior, label, dims_expr, dims_factor, noncentered, model):
    # NOTE: Do we really need this recursive function? Do we really allow for > 2 level hierarchies?
    kwargs = {}
    for key, value in prior.args.items():
        if isinstance(value, Prior):
            # hyperparam_key = term.hyperprior_alias.get(key, key) # FIXME?
            hyperparam_key = key
            hyperparam_label = f"{label}_{hyperparam_key}"
            kwargs[key] = _build_distribution(
                prior=value,
                label=hyperparam_label,
                dims_expr=dims_expr,
                dims_factor=tuple(),
                noncentered=noncentered,
                model=model,
            )
        else:
            kwargs[key] = value

    dims = dims_expr | dims_factor
    if noncentered and any(isinstance(v, pt.TensorVariable) for v in kwargs.values()):
        # non-centered is only relevant when distribution arguments are random variables.
        if prior.name == "Normal" and isinstance(kwargs.get("sigma", None), pt.TensorVariable):
            # TODO: Allow for 'mu' specification, too.
            sigma = kwargs["sigma"]
            offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims, model=model)
            return pm.Deterministic(label, offset * sigma, dims=dims, model=model)

        raise NotImplementedError(
            "The non-centered parametrization is only supported for Normal priors"
        )

    dist = get_distribution_from_prior(prior)
    return dist(label, **kwargs, dims=dims, model=model)
