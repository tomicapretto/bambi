import pymc as pm

MAPPING = {"Cumulative": pm.Categorical, "StoppingRatio": pm.Categorical}


def get_distribution(dist):
    """Return a PyMC distribution."""
    if isinstance(dist, str):
        if dist in MAPPING:
            dist = MAPPING[dist]
        elif hasattr(pm, dist):
            dist = getattr(pm, dist)
        else:
            raise ValueError(f"The Distribution '{dist}' was not found in PyMC")
    return dist


def get_distribution_from_prior(prior):
    if prior.dist is not None:
        distribution = prior.dist
    else:
        distribution = get_distribution(prior.name)
    return distribution
