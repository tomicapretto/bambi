import pymc as pm
from bambi.backend.pymc.utils import get_distribution_from_prior


def build_marginal_parameter(parameter, model):
    if isinstance(parameter.prior, (int, float)):
        return pm.Deterministic(parameter.label, parameter.prior, model=model)

    dims = tuple(
        model.__bambi_attrs__["response_coords_data"] | model.__bambi_attrs__["response_coords"]
    )
    dist = get_distribution_from_prior(parameter.prior)
    return dist(parameter.label, **parameter.prior.args, dims=dims)
