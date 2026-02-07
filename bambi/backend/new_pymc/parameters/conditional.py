import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps
import scipy as sp

from bambi.backend.new_pymc.terms import build_common_term, build_intercept_term
from bambi.config import config as bmb_config


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

    def build_group_specific(self):
        as_multivariate = False
        coefs = []
        columns = []
        predictors = []
        group_indexes = []

        # columns.append(term.data)
        # predictors.append(term.predictor)
        # group_indexes.append(term.group_index)

        if bmb_config["SPARSE_DOT"]:
            coefs_reshaped = []
            for coef in coefs:
                if as_multivariate and coef.ndim == 3:
                    # (f_j, e_j, k) -> (f_j * e_j, k)
                    coef_reshaped = coef.reshape(-1, coef.shape[-1])
                elif not as_multivariate and coef.ndim == 2:
                    # (f_j, e_j) -> (f_j * e_j,)
                    coef_reshaped = coef.flatten()
                else:
                    coef_reshaped = coef

                coefs_reshaped.append(coef_reshaped)

            # Design matrix Z: shape (n, q)
            data = sp.sparse.hstack(columns, format="csr")

            # Coefficients: shape (q, ) or (q, k)
            coefs = pt.concatenate(coefs_reshaped, axis=0)

            if not as_multivariate:
                coefs = coefs[:, np.newaxis]  # PyTensor expects 2D

            contribution = ps.structured_dot(data, coefs).squeeze()  # (n, ) or (n, K)
        else:
            for coef, predictor, group_index in zip(coefs, predictors, group_indexes):
                # The following code is short, but not simple.
                #
                # With multivariate models, we have:
                # When predictor.ndim > 1
                #     (n, e_j, k) * (n, e_j, 1) -> (n, e_j, k)
                #     (n, e_j, k).sum(1) -> (n, k)
                # Else
                #     (n, k) * (n, 1) -> (n, k)
                #
                # And with univariate models, we have:
                # When predictor.ndim > 1
                #     (n, e_j) * (n, e_j) -> (n, e_j)
                #     (n, e_j).sum(1) -> (n, )
                # Else
                #     (n, ) * (1, ) -> (n, )
                coef = coef[group_index]
                predictor_ndim = predictor.ndim

                if as_multivariate:
                    predictor = predictor[:, np.newaxis]

                term_contribution = coef * predictor

                if predictor_ndim > 1:
                    term_contribution = term_contribution.sum(axis=1)

                contribution += term_contribution

        for term in self.spec.group_specific_terms.values():
            group_specific_term = GroupSpecificTerm(term, bmb_model.noncentered)
            # Add coords
            for name, values in group_specific_term.coords.items():
                if name not in pymc_backend.model.coords:
                    pymc_backend.model.add_coords({name: values})

            coefs.append(group_specific_term.build(bmb_model))
            columns.append(term.data)
            predictors.append(term.predictor)
            group_indexes.append(term.group_index)

        if bmb_config["SPARSE_DOT"]:
            coefs_reshaped = []
            for coef in coefs:
                if as_multivariate and coef.ndim == 3:
                    # (f_j, e_j, k) -> (f_j * e_j, k)
                    coef_reshaped = coef.reshape(-1, coef.shape[-1])
                elif not as_multivariate and coef.ndim == 2:
                    # (f_j, e_j) -> (f_j * e_j,)
                    coef_reshaped = coef.flatten()
                else:
                    coef_reshaped = coef

                coefs_reshaped.append(coef_reshaped)

            # Design matrix Z: shape (n, q)
            data = sp.sparse.hstack(columns, format="csr")

            # Coefficients: shape (q, ) or (q, k)
            coefs = pt.concatenate(coefs_reshaped, axis=0)

            if not as_multivariate:
                coefs = coefs[:, np.newaxis]  # PyTensor expects 2D

            contribution = ps.structured_dot(data, coefs).squeeze()  # (n, ) or (n, k)
        else:
            contribution = 0
            for coef, predictor, group_index in zip(coefs, predictors, group_indexes):
                # The following code is short, but not simple.
                #
                # With multivariate models, we have:
                # When predictor.ndim > 1
                #     (n, e_j, k) * (n, e_j, 1) -> (n, e_j, k)
                #     (n, e_j, k).sum(1) -> (n, k)
                # Else
                #     (n, k) * (n, 1) -> (n, k)
                #
                # And with univariate models, we have:
                # When predictor.ndim > 1
                #     (n, e_j) * (n, e_j) -> (n, e_j)
                #     (n, e_j).sum(1) -> (n, )
                # Else
                #     (n, ) * (1, ) -> (n, )
                coef = coef[group_index]
                predictor_ndim = predictor.ndim

                if as_multivariate:
                    predictor = predictor[:, np.newaxis]

                term_contribution = coef * predictor

                if predictor_ndim > 1:
                    term_contribution = term_contribution.sum(axis=1)

                contribution += term_contribution

        # 'contribution' is of shape (n, ) or (n, k)
        self.output += contribution
