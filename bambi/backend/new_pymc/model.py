import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.new_pymc import coords
from bambi.backend.new_pymc.parameters import ConditionalParameter, MarginalParameter
from bambi.backend.new_pymc.coords import (
    coords_from_common,
    coords_from_group_specific,
    coords_from_response,
)

# NOTE: Should model be responsible of building parameters stuff?
#       Should dim/coordinate generation be independent of terms metadata generation?


class Model:
    def __init__(self, model):
        """_summary_

        Parameters
        ----------
        model : bmb.Model
            The bambi model specification.
        """
        self.model = None
        self.spec = model
        self.conditional_parameters = {
            key: ConditionalParameter(value)
            for key, value in self.spec.conditional_parameters.items()
        }
        self.marginal_parameters = {
            key: MarginalParameter(value) for key, value in self.spec.marginal_parameters.items()
        }

    def build(self):
        ## Global process:
        # 1. Build dims and coordinates
        # 2. Instantiate model
        # 3. Create data containers

        ## ConditionalParameter process
        # 1. Build RVs
        # 2. Manipulate data containers and RVs (grab stuff from PyMC model)
        # 3. Create deterministics (grab stuff from PyMC model)

        # 1. Build dims, coordinates, and term
        output_coords = coords_from_response(self.spec.response_term)
        output_coords.update(
            {
                "__obs__": np.arange(self.spec.response_term.data.shape[0]),
            }
        )
        model = pm.Model(coords=output_coords)
        model.__bambi_attrs__ = {
            "output_ndim": self.spec.family.output_ndim,
            "output_coords": output_coords,
        }

        for parameter in self.conditional_parameters.values():
            parameter.build(model)

        # 3. Build random variables

        # 4. Build expressions, grabbing them from the PyMC Model instance.

        # 5. Build response, again grabbing stuff from the PyMC model.

        self.response_coords = coords_from_response(self.spec.response_term)


def get_coords(self):
    coords = self.response_coords.copy()

    for parameter in self.conditional_parameters.values():
        for term in parameter.terms.values():
            coords.update(term.coords)

    for parameter in self.bambi_model.conditional_parameters.values():
        for term in parameter.common_terms.values():
            coords.update(coords_from_common(term))

        for term in parameter.group_specific_terms.values():
            coords.update(coords_from_group_specific(term))

    return coords
