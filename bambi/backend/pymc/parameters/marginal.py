import pymc as pm

from bambi.backend.pymc.utils import get_distribution_from_prior


def build_marginal_parameter(parameter, family, model: pm.Model):
    if isinstance(parameter.prior, (int, float)):
        return pm.Deterministic(parameter.label, parameter.prior, model=model)

    dims = tuple()
    param_spec = family.PARAMETERS[parameter.name]
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            dims = tuple(model.__bambi_attrs__["response_coords"])
        elif param_spec.coefs_dim == "response_reduced":
            dims = tuple(model.__bambi_attrs__["response_coords_reduced"])
        elif param_spec.coefs_dim == "response_cutpoints":
            dim_name = parameter.label + "_levels"
            response_levels = list(model.__bambi_attrs__["response_coords"].values())
            cutpoint_levels = []
            for l1, l2 in zip(response_levels[:-1], response_levels[1:]):
                cutpoint_levels.append(f"{l1}->{l2}")

            model.add_coords({dim_name: cutpoint_levels})
            dims = (dim_name,)

    dist = get_distribution_from_prior(parameter.prior)
    with model:
        rv = dist(parameter.label, **parameter.prior.args, dims=dims)
    return rv
