import pymc as pm
import pytensor.tensor as pt

from bambi.priors.prior import Prior
from bambi.backend.pymc.data import predictor_data_name, shape_common_data
from bambi.backend.pymc.types import Coords
from bambi.backend.pymc.utils import get_distribution_from_prior
from bambi.families.types import ParamSpec


def flatten_data(data: pt.Variable, coords: Coords) -> pt.Variable:
    if not coords:
        return data
    # The linear predictor is computed with dot(data, params),
    # so named term dimensions are flattened back into design-matrix columns.
    return data.reshape((data.shape[0], -1))


def build_smooth_coefficients(term, dims: tuple[str, ...], model: pm.Model) -> pt.Variable:
    name = term.label
    curved_count = term.shape[1] - term.null_space_dimension
    prior = term.prior

    constant = prior.get("constant", None)
    linear = prior["linear"]
    curvature = prior["curvature"]

    gaussian_case = True
    if constant and (not isinstance(constant, Prior) or constant.name != "Normal"):
        gaussian_case = False
    if not isinstance(linear, Prior) or linear.name != "Normal":
        gaussian_case = False
    if not isinstance(curvature, Prior) or curvature.name != "Normal":
        gaussian_case = False

    if gaussian_case:
        mu_blocks = []
        sigma_blocks = []

        if constant:
            mu_blocks.append(pt.atleast_1d(constant.args["mu"]))
            sigma_blocks.append(pt.atleast_1d(constant.args["sigma"]))

        mu_blocks.append(pt.atleast_1d(linear.args["mu"]))
        sigma_blocks.append(pt.atleast_1d(linear.args["sigma"]))

        curvature_sigma = curvature.args["sigma"]
        if isinstance(curvature_sigma, Prior):
            curvature_sigma = build_distribution(curvature_sigma, f"{name}_sigma", model)
        mu_blocks.append(pt.broadcast_to(curvature.args["mu"], (curved_count,)))
        sigma_blocks.append(pt.repeat(curvature_sigma, curved_count))

        mu = pt.concatenate(mu_blocks)
        sigma = pt.concatenate(sigma_blocks)
        with model:
            rv = pm.Normal(name, mu=mu, sigma=sigma, dims=dims)
        return rv

    # Handle other cases
    blocks = []
    if constant:
        constant_dist = build_distribution(constant, f"{name}_constant", model)
        blocks.append(constant_dist)

    linear_dist = build_distribution(linear, f"{name}_linear", model)
    curvature_dist = build_distribution(curvature, f"{name}_curvature", model)
    blocks.append(linear_dist)
    blocks.append(curvature_dist)

    with model:
        rv = pm.Deterministic(
            name, pt.concatenate([pt.atleast_1d(block) for block in blocks]), dims=dims
        )

    return rv


def build_distribution(prior: Prior, label: str, model: pm.Model) -> pt.Variable:
    kwargs = {}
    for name, value in prior.args.items():
        if isinstance(value, Prior):
            kwargs[name] = build_distribution(prior=value, label=f"{label}_{name}", model=model)
        else:
            kwargs[name] = value

    dist = get_distribution_from_prior(prior)
    with model:
        rv = dist(label, **kwargs)

    return rv


def build_smooth_term(
    term_info, param_spec: ParamSpec, model: pm.Model
) -> tuple[pt.Variable, pt.Variable]:
    term = term_info.term
    param_name = term.label
    coords = term_info.coords
    data_name = predictor_data_name(term.label, term_info.data_dims, model)

    # Register coords
    if data_name not in model or param_name not in model:
        model.add_coords(coords)

    # Register data
    if data_name not in model:
        data = shape_common_data(term.data, coords)
        pm.Data(data_name, data, dims=term_info.data_dims, model=model)

    # Register parameter
    response_coords = {}
    if param_spec.ndim > 0:
        if param_spec.coefs_dim == "response":
            response_coords = model.__bambi_attrs__["response_coords"]
        elif param_spec.coefs_dim == "response_reduced":
            response_coords = model.__bambi_attrs__["response_coords_reduced"]

    if response_coords:
        raise NotImplementedError("Smooths require scalar response parameters.")

    param_coords = coords | response_coords
    param_dims = tuple(param_coords)
    param = build_smooth_coefficients(term, param_dims, model)

    return flatten_data(model[data_name], coords), param
