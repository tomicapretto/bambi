from bambi.priors.prior import Prior
from bambi.terms.common import CommonTerm


class SmoothTerm(CommonTerm):
    """Common term with one Normal vector and a hierarchical curvature scale.

    The ``null`` prior provides fixed Normal means and scales for the leading
    null-space coefficients. The ``sigma`` prior provides one shared random
    scale for all remaining coefficients, in the same spline dimension.
    """

    @property
    def null_space_dimension(self):
        return self.components[0].call.stateful_transform.null_space_dimension

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        defaults = {
            "null": Prior("Normal", mu=0, sigma=2.5),
            "sigma": Prior("HalfNormal", sigma=1),
        }
        if value is not None:
            if not isinstance(value, dict) or set(value) - set(defaults):
                raise ValueError("Smooth priors must be a dictionary with 'null' and/or 'sigma'.")
            if not all(isinstance(prior, Prior) for prior in value.values()):
                raise ValueError("Smooth priors must contain Prior instances.")
            defaults.update(value)
        if any(
            isinstance(arg, Prior) for prior in defaults.values() for arg in prior.args.values()
        ):
            raise ValueError("Nested hyperpriors are not supported for smooth priors.")
        null = defaults["null"]
        if null.name != "Normal" or null.dist is not None or set(null.args) != {"mu", "sigma"}:
            raise ValueError("The null-space prior must be Normal with 'mu' and 'sigma'.")
        self._prior = defaults
