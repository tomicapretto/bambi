import numpy as np
import pytensor.tensor as pt

from bambi.backend.materialize.utils import get_distribution_from_prior
from bambi.families.multivariate import MultivariateFamily, Categorical


def materialize_intercept(term, model):
    """Build term.

    Parameters
    ----------
    term : bambi.terms.Term
        An object representing the intercept.
    spec : bambi.Model
        The model instance.

    Returns
    -------
    dist : pm.Distribution
        A PyMC distribution of shape `(1, )` or `(1, K)`.
    """
    distribution = get_distribution_from_prior(term.prior)

    if isinstance(model.family, (MultivariateFamily, Categorical)):
        # shape: (1, K)
        dims = list(model.response_component.term.coords)
        dist = distribution(term.name, dims=dims, **term.prior.args)[np.newaxis, :]
    else:
        # shape: (1,)
        dist = pt.atleast_1d(distribution(term.name, **term.prior.args))

    return dist
