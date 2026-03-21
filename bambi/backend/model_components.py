import numpy as np

from bambi.backend.utils import get_distribution_from_prior
from bambi.families.univariate import Cumulative, StoppingRatio


ORDINAL_FAMILIES = (Cumulative, StoppingRatio)


class ConstantComponent:
    def __init__(self, component):
        self.component = component
        self.output = 0

    def build(self, pymc_backend, bmb_model):
        label = self.component.alias if self.component.alias else self.component.name

        # NOTE: This could be handled in a different manner in the future, only applies to
        # thresholds and assumes we always do it when using ordinal families.
        extra_args = {}
        if isinstance(bmb_model.family, ORDINAL_FAMILIES):
            threshold_dim = label + "_dim"
            threshold_values = np.arange(len(bmb_model.response_component.term.levels) - 1)
            extra_args["dims"] = threshold_dim
            pymc_backend.model.add_coords({threshold_dim: threshold_values})

        with pymc_backend.model:
            if isinstance(self.component.prior, (int, float)):
                # Set to a constant value
                self.output = self.component.prior
            else:
                # Set to a distribution
                dist = get_distribution_from_prior(self.component.prior)
                self.output = dist(label, **self.component.prior.args, **extra_args)
