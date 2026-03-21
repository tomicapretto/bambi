import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
import scipy as sp

from bambi.backend.new_pymc.terms import (
    build_common_term,
    build_intercept_term,
    build_group_specific_term_dot,
)
from bambi.config import config as bmb_config


class ConditionalParameter:
    """Deterministic parameter computed as a function of data and other parameters."""

    def __init__(self, parameter):
        self.spec = parameter

    def build(self, model):
        self.value = 0
        if self.spec.intercept_term:
            self.value += self.build_intercept(model)

        if self.spec.common_terms:
            self.value += self.build_common(model)

        if self.spec.group_specific_terms:
            self.value += self.build_group_specific(model)

    def build_intercept(self, model):
        if model.__bambi_attrs__["output_ndim"] == 1:
            ensure_ndim = pt.atleast_1d
        else:
            ensure_ndim = pt.atleast_2d

        return ensure_ndim(build_intercept_term(self.spec.intercept_term, model))

    def build_common(self, model):
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
        data = pt.concatenate(data_list, axis=1)  # (n, p)

        # TODO: Use deterministic if we center covariates.
        # 'pt.dot(data, params)' is of shape (n, ) or (n, K)
        return pt.dot(data, params)

    def build_group_specific(self, model):
        if bmb_config["SPARSE_DOT"]:
            return self._build_group_specific_dot(self, model)
        return self._build_group_specific_idx(self, model)

    def _build_group_specific_dot(self, model):
        data_blocks = []
        param_blocks = []
        for term in self.spec.group_specific_terms.values():
            data, param = build_group_specific_term_dot(term, model)
            data_blocks.append(data)
            param_blocks.append(param)

        # Design matrix Z: shape (n, q)
        data = sp.sparse.hstack(data_blocks, format="csr")

        # Coefficients array: shape (q, ) or (q, K)
        coefs = pt.concatenate(param_blocks, axis=0)

        if coefs.ndim == 1:
            # PyTensor expects 2D
            coefs = coefs[:, np.newaxis]

        # (n, ) or (n, K)
        # FIXME: Do we always need to squeeze?
        return ps.structured_dot(data, coefs).squeeze()

    def _build_group_specific_idx(self, model):
        contribution = 0
        for term in self.spec.group_specific_terms.values():
            contribution += build_group_specific_term_dot(term, model)
        return contribution
