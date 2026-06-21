import pymc as pm

from bambi.backend.pymc.coords import coords_from_response
from bambi.backend.pymc.parameters import build_conditional_parameter, build_marginal_parameter
from bambi.backend.pymc.terms import build_response_term


class PyMCModel:
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
        # 3. Build random variables
        # 4. Build response, again grabbing stuff from the PyMC model.

        response_coords_data, response_coords, response_coords_reduced = coords_from_response(
            self.spec.response_term, self.spec.family
        )

        model = pm.Model(coords=response_coords_data | response_coords | response_coords_reduced)
        model.__bambi_attrs__ = {
            "response_ndim": self.spec.family.RESPONSE_NDIM,
            "response_coords_data": response_coords_data,
            "response_coords": response_coords,
            "response_coords_reduced": response_coords_reduced,
        }

        marginal_parameters = {}
        conditional_parameters = {}
        for name, parameter in self.spec.marginal_parameters.items():
            marginal_parameters[name] = build_marginal_parameter(parameter, model)

        for name, parameter in self.spec.conditional_parameters.items():
            conditional_parameters[name] = build_conditional_parameter(
                parameter, self.spec.family, model
            )

        build_response_term(
            term=self.spec.response_term,
            parameters=marginal_parameters | conditional_parameters,
            family=self.spec.family,
            model=model,
        )

        self.model = model

    def run(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,  # pylint: disable=unused-argument
        include_response_params=False,  # pylint: disable=unused-argument
        inference_method="pymc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        if inference_method != "pymc":
            raise NotImplementedError("Only inference_method='pymc' is currently supported")

        with self.model:
            output = pm.sample(
                draws=draws,
                tune=tune,
                discard_tuned_samples=discard_tuned_samples,
                init=init,
                n_init=n_init,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                **kwargs,
            )
        return output
