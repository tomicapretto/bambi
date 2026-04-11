import pytensor.tensor as pt

from bambi.backend.new_pymc.transform.register import (
    transform_general_predictor,
    manipulate_data,
    manipulate_parameters,
)

from bambi.families.univariate import (
    Beta,
    BetaBinomial,
    Binomial,
    Categorical,
    Cumulative,
    Exponential,
    Gamma,
    HurdleGamma,
    StoppingRatio,
    Weibull,
    ZeroInflatedBinomial,
)

from bambi.families.multivariate import Multinomial, DirichletMultinomial


@manipulate_parameters(Beta)
def _(parameters):
    mu = parameters["mu"]
    kappa = parameters["kappa"]
    return {"alpha": mu * kappa, "beta": (1 - mu) * kappa}


@manipulate_data(Binomial)
def _(data):
    return {"observed": data[:, 0], "n": data[:, 1]}


@manipulate_parameters(BetaBinomial)
def _(parameters):
    mu = parameters["mu"]
    kappa = parameters["kappa"]
    return {"alpha": mu * kappa, "beta": (1 - mu) * kappa}


@manipulate_data(BetaBinomial)
def _(data):
    return {"observed": data[:, 0], "n": data[:, 1]}


@transform_general_predictor(Categorical, "p")
def _(predictor, parameters):
    if predictor.ndim == 1:
        zeros = pt.zeros(shape=(1,))
    else:
        zeros = pt.zeros(shape=(predictor.shape[0], 1))
    return pt.concatenate((zeros, predictor), axis=-1)


@transform_general_predictor(Cumulative, "p")
def _(predictor, parameters):
    threshold = parameters["threshold"]

    if predictor == 0:
        # When the model does not have any predictors. PyMC will reshape accordingly.
        predictor = threshold
    else:
        predictor = threshold - pt.shape_padright(predictor)

    return predictor


@manipulate_parameters(Cumulative)
def _(parameters):
    return {"p": parameters["p"]}


@manipulate_data(DirichletMultinomial)
def _(data):
    return {"observed": data, "n": data.sum(axis=1).astype(int)}


@manipulate_parameters(Exponential)
def _(parameters):
    return {"lam": 1 / parameters["mu"]}


@manipulate_parameters(Gamma)
def _(parameters):
    return {
        "mu": parameters["mu"],
        "sigma": parameters["mu"] / (parameters["alpha"] ** 0.5),
    }


@manipulate_parameters(HurdleGamma)
def _(parameters):
    return {
        "mu": parameters["mu"],
        "sigma": parameters["mu"] / (parameters["alpha"] ** 0.5),
        "psi": parameters["psi"],
    }


@transform_general_predictor(Multinomial, "p")
def _(predictor, parameters):
    if predictor.ndim == 1:
        zeros = pt.zeros(shape=(1,))
    else:
        zeros = pt.zeros(shape=(predictor.shape[0], 1))
    return pt.concatenate((zeros, predictor), axis=-1)


@manipulate_data(Multinomial)
def _(data):
    return {"observed": data, "n": data.sum(axis=1).astype(int)}


@transform_general_predictor(StoppingRatio, "p")
def _(predictor, parameters):
    threshold = parameters["threshold"]

    if predictor == 0:
        # When the model does not have any predictors, PyMC reshapes things accordingly.
        predictor = threshold
    else:
        predictor = threshold - pt.shape_padright(predictor)

    return predictor


@manipulate_parameters(StoppingRatio)
def _(parameters):
    return {"p": parameters["p"]}


@manipulate_parameters(Weibull)
def _(parameters):
    mu = parameters["mu"]
    alpha = parameters["alpha"]
    return {
        "alpha": alpha,
        "beta": mu / pt.gamma(1 + 1 / alpha),
    }


@manipulate_data(ZeroInflatedBinomial)
def _(data):
    return {"observed": data[:, 0], "n": data[:, 1]}
