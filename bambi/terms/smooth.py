import warnings

import formulae

from bambi.priors import Prior
from bambi.terms.base import BaseTerm


class SmoothTerm(BaseTerm):
    """Term for a penalized smooth with a hierarchical curvature prior."""

    def __init__(self, term, prior, prefix=None):
        self.term = term
        self.prior = prior
        self.data = term.data
        self.prefix = prefix

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def kind(self):
        return self.term.kind

    @property
    def shape(self):
        return self.data.shape

    @property
    def levels(self):
        return None

    @property
    def categorical(self):
        return False

    @property
    def basis(self):
        return self.components[0].call.stateful_transform.__transform_name__

    @property
    def null_space_dimension(self):
        return self.components[0].call.stateful_transform.null_space_dimension

    @property
    def has_intercept(self):
        return self.null_space_dimension == 2

    @property
    def prior_keys(self):
        keys = {"linear", "curvature"}
        if self.has_intercept:
            keys.add("constant")
        return keys

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        if value is None:
            return

        if not isinstance(value, dict) or set(value) - self.prior_keys:
            raise ValueError(
                f"Smooth priors must be a dictionary with keys in {sorted(self.prior_keys)}."
                f"Currently has keys {sorted(value)}."
            )

        curvature_prior = value["curvature"]
        if curvature_prior.name == "Normal":
            if isinstance(curvature_prior.args["mu"], dict):
                warnings.warn(
                    "The curvature prior usually has a scalar mean.",
                    UserWarning,
                    stacklevel=2,
                )

            if not isinstance(curvature_prior.args["sigma"], Prior):
                warnings.warn(
                    "The scale of the curvature prior is usually modeled as a random variable.",
                    UserWarning,
                    stacklevel=2,
                )

        self._prior = value

    def __str__(self):
        return self.make_str()
