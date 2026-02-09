import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.new_pymc.terms import build_common_term, build_intercept_term


class ConditionalParameter:
    """Deterministic parameter computed as a function of data and other parameters."""

    def __init__(self, parameter):
        self.spec = parameter

    def build(self, model):
        self.output = 0
        if self.spec.intercept_term:
            self.output += self.build_intercept(model)

        if self.spec.common_terms:
            self.output += self.build_common(model)

        if self.spec.group_specific_terms:
            self.output += self.build_group_specific(model)

    def build_intercept(self, model):
        if model.__bambi_attrs__["output_ndim"] == 1:
            ensure_ndim = pt.atleast_1d
        else:
            ensure_ndim = pt.atleast_2d

        return ensure_ndim(build_intercept_term(self.spec.intercept_term, model))

    def build_common(self, model):
        if not self.spec.common_terms:
            return 0

        data_list = []
        param_list = []

        if model.__bambi_attrs__["output_ndim"] == 1:
            ensure_ndim = pt.atleast_1d
        else:
            ensure_ndim = pt.atleast_2d

        for term in self.spec.common_terms.values():
            data, param = build_common_term(term, model)

            data_list.append(data)
            param_list.append(ensure_ndim(param))

        params = pt.concatenate(param_list, axis=0)  # (p, ) or (p, K)
        data = np.column_stack(data_list)  # (n, p)

        model.add_coord(f"{self.spec.name}_data_dim_1", np.arange(data.shape[1]))
        data = pm.Data(
            f"{self.spec.name}_data", data, dims=("__obs__", f"{self.spec.name}_data_dim_1")
        )

        # 'pt.dot(data, params)' is of shape (n, ) or (n, K)
        return pt.dot(data, params)

    def build_group_specific(self):
        pass
