import numpy as np


class ResponseTerm:
    def __init__(self, term, family):
        self.term = term
        self.family = family

    def robustify_dims(self, pymc_backend, kwargs):
        # It's possible the observed for the response is multidimensional,
        # but there's a single linear predictor because the family is not multivariate.
        # In this case, we add extra dimensions to avoid having shape mismatch between the data
        # and the shape implied by the `dims` we pass.

        if (
            self.term.is_censored
            or self.term.is_truncated
            or self.term.is_weighted
            or self.term.is_constrained
        ):
            return kwargs

        dims, data = kwargs["dims"], kwargs["observed"]
        dims_n = len(dims)
        ndim_diff = data.ndim - dims_n

        # TO DO: Test with multinomial regression, shouldn't be added?
        if ndim_diff > 0:
            for i in range(ndim_diff):
                axis = dims_n + i
                name = f"{self.name}_extra_dim_{i}"
                values = np.arange(np.size(data, axis=axis))
                pymc_backend.model.add_coords({name: values})
                dims = dims + (name,)
        kwargs["dims"] = dims
        return kwargs
