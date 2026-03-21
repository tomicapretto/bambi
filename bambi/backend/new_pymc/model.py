import pymc as pm

from bambi.backend.new_pymc.coords import coords_from_response
from bambi.backend.new_pymc.parameters import build_conditional_parameter, build_marginal_parameter
from bambi.backend.new_pymc.terms import build_response_term


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
        model = pm.Model(coords=output_coords)
        model.__bambi_attrs__ = {
            "output_ndim": self.spec.family.output_ndim,
            "output_coords": output_coords,
        }

        # 3. Build random variables
        marginal_parameters = {}
        conditional_parameters = {}
        for name, parameter in self.marginal_parameters.items():
            marginal_parameters[name] = build_marginal_parameter(parameter, self.spec.family, model)

        for name, parameter in self.conditional_parameters.items():
            conditional_parameters[name] = build_conditional_parameter(
                parameter, self.spec.family, model
            )

        # 4. Build response, again grabbing stuff from the PyMC model.
        parameters = marginal_parameters | conditional_parameters

        build_response_term(
            term=self.spec.response_term,
            parameters=parameters,
            family=self.spec.family,
            model=model,
        )

        self.model = model
