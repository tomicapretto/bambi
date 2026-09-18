import numpy as np
import pandas as pd
import pymc as pm
import pytest

from formulae.transforms import NaturalCubicSpline
from scipy.stats import norm
from xarray import DataTree

import bambi as bmb
from bambi.terms.smooth import SmoothTerm
from bambi.transformations import CRSpline


@pytest.fixture
def smooth_data():
    x = np.linspace(0, 1, 40)
    return pd.DataFrame({"x": x, "y": np.sin(6 * x)})


@pytest.mark.parametrize("center", [True, False])
def test_random_basis_and_prediction(smooth_data, center):
    original, adapted = NaturalCubicSpline(), CRSpline()
    original(smooth_data.x, df=6, center=center)
    np.testing.assert_allclose(adapted(smooth_data.x, df=6, center=center), original.to_random())
    new_x = np.array([-0.2, 0.4, 1.3])
    np.testing.assert_allclose(adapted.eval(new_x), original.to_random(original.eval(new_x)))
    assert adapted.null_space_dimension == 2 - int(center)


def test_hierarchical_prior_and_new_data(smooth_data):
    name = "cr(x, df=6)"
    model = bmb.Model(f"y ~ {name}", smooth_data)
    term = model.parameters["mu"].terms[name]
    assert isinstance(term, SmoothTerm)
    assert term.prior["curvature"].args["sigma"].name == "HalfNormal"
    model.set_alias({name: "smooth"})
    model.build()
    pymc_model = model.backend.model
    assert pymc_model["smooth"] in pymc_model.free_RVs
    assert pymc_model.named_vars_to_dims["smooth"] == (f"{name}_dim",)
    assert not any(dim.endswith(("_null_dim", "_curved_dim")) for dim in pymc_model.coords)
    assert "smooth_offset" not in pymc_model
    values = np.linspace(-1, 1, 6)
    logp = pm.logp(pymc_model["smooth"], values)
    np.testing.assert_allclose(
        logp.eval({pymc_model["smooth_curvature_sigma"]: 0.75}),
        norm.logpdf(values, scale=[2.5, *([0.75] * 5)]),
    )
    with model.backend.model:
        prior = pm.sample_prior_predictive(draws=10, random_seed=42)
    # Use prior draws as coefficient draws to check the prediction path without MCMC.
    draws = DataTree.from_dict({"posterior": prior["prior"].to_dataset()})
    new_data = pd.DataFrame({"x": [-0.2, 0.1, 0.8, 1.2]})
    prediction = model.predict(draws, data=new_data, inplace=False)
    posterior = prediction["posterior"]
    basis = term.term.eval_new_data(new_data)
    expected = np.einsum("np,cdp->cdn", basis, posterior["smooth"].values)
    expected += posterior["Intercept"].values[..., None]
    np.testing.assert_allclose(prediction["predictions"]["mu"].values, expected, atol=1e-10)


def test_custom_priors(smooth_data):
    name = "cr(x, df=6)"
    model = bmb.Model(
        f"y ~ {name}",
        smooth_data,
        priors={
            name: {"curvature": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=3))}
        },
    )
    assert model.parameters["mu"].terms[name].prior["curvature"].args["sigma"].args["sigma"] == 3
    model.build()


def test_uncentered_smooth(smooth_data):
    name = "cr(x, df=6, center=False)"
    with pytest.raises(ValueError, match="center=True"):
        bmb.Model(f"y ~ {name}", smooth_data)
    model = bmb.Model(
        f"y ~ 0 + {name}",
        smooth_data,
        priors={
            name: {
                "constant": bmb.Prior("Normal", mu=1, sigma=3),
                "linear": bmb.Prior("Normal", mu=2, sigma=4),
            }
        },
    )
    assert model.parameters["mu"].terms[name].null_space_dimension == 2
    model.build()
    pymc_model = model.backend.model
    values = np.linspace(-1, 1, 6)
    np.testing.assert_allclose(
        pm.logp(pymc_model[name], values).eval({pymc_model[f"{name}_curvature_sigma"]: 0.75}),
        norm.logpdf(values, loc=[1, 2, 0, 0, 0, 0], scale=[3, 4, 0.75, 0.75, 0.75, 0.75]),
    )


def test_non_normal_component_prior_rejected(smooth_data):
    name = "cr(x, df=6)"
    with pytest.raises(ValueError, match="'linear' prior must be Normal"):
        bmb.Model(
            f"y ~ {name}", smooth_data, priors={name: {"linear": bmb.Prior("Laplace", mu=0, b=1)}}
        )


def test_smooth_interaction_rejected(smooth_data):
    with pytest.raises(NotImplementedError, match="Interactions"):
        bmb.Model("y ~ x:cr(x, df=6)", smooth_data)


def test_group_specific_smooth_rejected(smooth_data):
    smooth_data["group"] = np.tile(["a", "b"], 20)
    with pytest.raises(NotImplementedError, match="Group-specific smooths"):
        bmb.Model("y ~ (cr(x, df=6)|group)", smooth_data)
