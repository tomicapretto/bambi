import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.new_pymc.utils import get_distribution_from_prior
from bambi.priors import Prior  # TODO: remove?


def exp_quad(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.ExpQuad(input_dim, ls=ell)


def matern32(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern32(input_dim, ls=ell)


def matern52(sigma, ell, input_dim=1):
    return sigma**2 * pm.gp.cov.Matern52(input_dim, ls=ell)


GP_KERNELS = {
    "ExpQuad": {"fn": exp_quad, "params": ("sigma", "ell")},
    "Matern32": {"fn": matern32, "params": ("sigma", "ell")},
    "Matern52": {"fn": matern52, "params": ("sigma", "ell")},
}


def build_hsgp_term(term, model):
    """_summary_

    Parameters
    ----------
    term : bambi.terms.HSGPTerm
        write me
    model : pymc.Model
        write me
    """
    covariance_functions = build_covariance_function(term, model)

    # Prepare dims
    coeff_dims = (f"{term.name}_weights_dim",)
    contribution_dims = ("__obs__",)

    # Data may be scaled so the maximum Euclidean distance between two points is 1
    if term.scale_predictors:
        data = term.data_centered / term.maximum_distance
    else:
        data = term.data_centered

    # Build HSGP and store it in the term.
    if term.by_levels is not None:
        coeff_dims = coeff_dims + (f"{term.name}_by",)
        phi_list, sqrt_psd_list = [], []
        term.hsgp = {}

        # Because of the filter in the loop, it will be as if the observations were sorted
        # using the values of the 'by' variable.
        # This approach helps especially when there are many groups, which causes many zeros
        # with other approaches (until PyMC and us have better support for sparse matrices)
        indexes_to_unsort = term.by.argsort(kind="mergesort").argsort(kind="mergesort")
        for i, level in enumerate(term.by_levels):
            cov_func = covariance_functions[i]
            # Notes:
            # 'm' doesn't change by group
            # We need to use list() in 'm' and 'L' because arrays are not instance of Sequence
            hsgp = pm.gp.HSGP(
                m=list(term.m),
                L=list(term.L[i]),
                drop_first=term.drop_first,
                cov_func=cov_func,
            )
            # Notice we pass all the values, for all the groups.
            # Then we only keep the ones for the corresponding group.
            phi, sqrt_psd = hsgp.prior_linearized(data[term.by == i])
            sqrt_psd_list.append(sqrt_psd)
            phi_list.append(phi.eval())

            # Store it for later usage
            term.hsgp[level] = hsgp
        sqrt_psd = pt.stack(sqrt_psd_list, axis=1)
    else:
        (cov_func,) = covariance_functions
        term.hsgp = pm.gp.HSGP(
            m=list(term.m),
            L=list(term.L[0]),
            drop_first=term.drop_first,
            cov_func=cov_func,
        )
        # Get prior components
        phi, sqrt_psd = term.hsgp.prior_linearized(data)
        phi = phi.eval()

    # Build weights coefficient
    # Handle the case where the outcome is multivariate
    if model.__bambi_attrs__["output_ndim"] == 2:
        # Append the dims of the response variables to the coefficient and contribution dims
        # In general:
        # coeff_dims: ('weights_dim', ) -> ('weights_dim', f'{response}_dim')
        # contribution_dims: ('__obs__', ) -> ('__obs__', f'{response}_dim')
        response_dims = tuple(model.__bambi_attrs__["output_coords"])
        coeff_dims = coeff_dims + response_dims
        contribution_dims = contribution_dims + response_dims

        # Append a dimension to sqrt_psd: ('weights_dim', ) -> ('weights_dim', 1)
        sqrt_psd = sqrt_psd[:, np.newaxis]

    if term.centered:
        coeffs = pm.Normal(f"{term.name}_weights", sigma=sqrt_psd, dims=coeff_dims)
    else:
        coeffs_raw = pm.Normal(f"{term.name}_weights_raw", dims=coeff_dims)
        coeffs = pm.Deterministic(f"{term.name}_weights", coeffs_raw * sqrt_psd, dims=coeff_dims)

    # Build deterministic for the HSGP contribution
    # If there are groups, we do as many dot products as groups
    if term.by_levels is not None:
        contribution_list = []
        for i in range(len(term.by_levels)):
            contribution_list.append(phi_list[i] @ coeffs[:, i])
        # We need to unsort the contributions so they match the original data
        contribution = pt.concatenate(contribution_list)[indexes_to_unsort]
    # If there are no groups, it's a single dot product
    else:
        contribution = pt.dot(phi, coeffs)  # "@" operator is not working as expected

    return pm.Deterministic(term.name, contribution, dims=contribution_dims, model=model)


def build_covariance_function(term, model):
    cov_dict = GP_KERNELS[term.cov]
    create_covariance_function = cov_dict["fn"]
    param_names = cov_dict["params"]
    params = {}

    # Set dimensions and behavior for priors that are actually fixed (floats or ints)
    if term.by_levels is not None and not term.share_cov:
        dims = (f"{term.name}_by",)
        recycle = True
    else:
        dims = tuple()
        recycle = False

    # Build priors and parameters
    for param_name in param_names:
        prior = term.prior[param_name]
        param_dims = dims
        if isinstance(prior, Prior):
            dist = get_distribution_from_prior(prior)
            # varying lengthscale parameter
            if param_name == "ell" and not term.iso and term.shape[1] > 1:
                param_dims = (f"{term.name}_var",) + param_dims
            value = dist(f"{term.name}_{param_name}", **prior.args, dims=param_dims, model=model)
        else:
            # The value is constant
            if recycle:
                value = (prior,) * term.groups_n
            else:
                value = prior

        params[param_name] = value

    if not term.share_cov:
        # squeeze makes sure the array is 0d when term.groups_n is 1
        params["input_dim"] = np.repeat(term.shape[1], term.groups_n).squeeze()

    if term.groups_n == 1 or term.share_cov:
        # All groups use the same covariance function
        covariance_function = create_covariance_function(**params)
        output = [covariance_function] * term.groups_n
    else:
        # Each group gets its own covariance function
        output = []
        for i in range(len(term.by_levels)):
            params_level = {}
            # FIXME: How can we be sure `value` has the correct dimension?
            #        `value` has to be an ndarray for .ndim to work, the second check is nonsensical
            for key, value in params.items():
                if value[..., i].ndim == 0 and isinstance(value, np.ndarray):
                    entry = value[..., i].item()
                else:
                    entry = value[..., i]
                params_level[key] = entry
            covariance_function = create_covariance_function(**params_level)
            output.append(covariance_function)

    return output
