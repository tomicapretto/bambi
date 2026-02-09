import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.new_pymc.coords import (
    coords_from_common,
    coords_from_group_specific,
    coords_from_response,
)

# NOTE: Should model be responsible of building parameters stuff?
#       Should dim/coordinate generation be independent of terms metadata generation?


class Model:
    def __init__(self, model):
        self.pymc_model = None
        self.bambi_model = model
        self._response_coords = coords_from_response(model.response_term)
        self.conditional_parameters = {
            key: ConditionalParameter(value) for key, value in model.conditional_parameters.items()
        }
        self.marginal_parameters = {
            key: MarginalParameter(value) for key, value in model.marginal_parameters.items()
        }

    def get_coords(self):
        coords = self._response_coords.copy()

        for parameter in self.bambi_model.conditional_parameters.values():
            for term in parameter.common_terms.values():
                coords.update(coords_from_common(term))

            for term in parameter.group_specific_terms.values():
                coords.update(coords_from_group_specific(term))

        return coords

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
        # for parameter in self.bambi_model.marginal_parameters:
        #     ...
        coords = self.get_coords()
        model = pm.Model(coords=coords)

        with pm.Model(coords=coords) as model:
            # 2. Build data containers -> Has
            for parameter in self.bambi_model.conditional_parameters.values():
                # for term in parameter
                pass

        self.pymc_model = model

        # 3. Build random variables

        # 4. Build expressions, grabbing them from the PyMC Model instance.

        # 5. Build response, again grabbing stuff from the PyMC model.


class CommonTerm:
    def __init__(self, term):
        self.term = term
        self.coords = coords_from_common(self.term)
        self.data_name = f"{self.term.name}_data"
        self.data_dims = None
        self.param_name = None

    def register_data(self, model):
        if self.data_name in model:
            return None

        output_ndim = model.__bambi_attrs__["output_ndim"]
        self.data_dims = ["__obs__", *self.coords, *model.__bambi_attrs__["output_coords"]]

        if output_ndim == 1:
            data_shape = (self.term.data.shape[0], -1)
        elif output_ndim == 2:
            data_shape = (self.term.data.shape[0], -1, self.term.data.shape[-1])
        else:
            raise ValueError("panic")

        data = np.reshape(self.term.data, data_shape)
        pm.Data(self.data_name, data, dims=self.data_dims, model=model)

    def register_parameter(self, model):
        # NOTE:
        #   When repsonse_ndim == 2 we need to use `pt.atleast_2d`
        #   Otherwise, we use `pt.atleast_1d`.
        # NOTE:
        #   Do we check if it's in the model before adding it?
        # NOTE:
        #   When len(term.coords) > 1, we need to reshape.
        #   When output_ndim == 1: x.reshape(-1)
        #   When output_ndim == 2: x.reshape(-1, x.shape[-1])
        coords = self.coords | model.__bambi_attrs__["output_coords"]
        dims = tuple(coords)
        shape = tuple(len(coord) for coord in coords.values())

        kwargs = {
            name: np.broadcast_to(value, shape) for name, value in self.term.prior.args.items()
        }

        dist = get_distribution_from_prior(self.term.prior)
        dist(self.param_name, **kwargs, dims=dims, model=model)


class MarginalParameter:
    """Global, free parameter shared across observations."""


class ConditionalParameter:
    """Deterministic parameter computed as a function of data and other parameters."""

    def __init__(self, parameter):
        self.bambi_parameter = parameter

    def build(self, model):
        self.build_common(model)

    def build_common(self, model):
        terms = []

    def build_group_specific(self):
        pass


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
