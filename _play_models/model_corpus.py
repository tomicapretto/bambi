"""Small local corpus for driving the parameter/backend migration.

This file is intentionally gitignored. It is a runnable scratch harness that
keeps representative model specifications close to the repo while avoiding test
suite churn during the refactor.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

import bambi as bmb


RNG_SEED = 121195


@dataclass(frozen=True)
class Case:
    name: str
    formula: str | bmb.Formula
    data: Callable[[], pd.DataFrame]
    family: str = "gaussian"
    kwargs: dict | None = None

    def make_model(self) -> bmb.Model:
        kwargs = {} if self.kwargs is None else self.kwargs
        return bmb.Model(self.formula, self.data(), family=self.family, **kwargs)


def gaussian_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    condition = rng.choice(["ctl", "trt"], size=n)
    subject = rng.choice([f"s{i}" for i in range(6)], size=n)
    item = rng.choice([f"i{i}" for i in range(8)], size=n)
    y = (
        1.0
        + 0.7 * x
        - 0.25 * z
        + np.where(group == "b", 0.35, 0)
        + np.where(condition == "trt", -0.25, 0)
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "group": group,
            "condition": condition,
            "subject": subject,
            "item": item,
        }
    )


def spline_data(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = np.linspace(-2.5, 2.5, n)
    z = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    subject = rng.choice([f"s{i}" for i in range(6)], size=n)
    y = 0.5 + np.sin(1.5 * x) + 0.3 * z + np.where(group == "c", 0.4, 0)
    y = y + rng.normal(scale=0.35, size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "group": group, "subject": subject})


def binary_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    eta = -0.25 + 0.8 * x + np.where(group == "b", 0.5, 0)
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p, size=n)
    return pd.DataFrame({"y": y, "x": x, "group": group})


def count_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    mu = np.exp(0.4 + 0.35 * x + np.where(group == "c", 0.25, 0))
    y = rng.poisson(mu)
    return pd.DataFrame({"y": y, "x": x, "group": group})


def gamma_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    mu = np.exp(0.5 + 0.4 * x + np.where(group == "b", -0.2, 0))
    alpha = 2.5
    y = rng.gamma(shape=alpha, scale=mu / alpha)
    return pd.DataFrame({"y": y, "x": x, "group": group})


def beta_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    group = rng.choice(["a", "b", "c"], size=n)
    eta = -0.15 + 0.5 * x + np.where(group == "c", 0.3, 0)
    mu = 1 / (1 + np.exp(-eta))
    kappa = 12
    y = rng.beta(mu * kappa, (1 - mu) * kappa)
    return pd.DataFrame({"y": y, "x": x, "group": group})


def categorical_response_data(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=n)
    group = rng.choice(["a", "b"], size=n)
    logits = np.column_stack(
        [
            np.zeros(n),
            -0.2 + 0.4 * x + np.where(group == "b", 0.3, 0),
            0.1 - 0.3 * x + np.where(group == "b", -0.2, 0),
        ]
    )
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)
    y = [rng.choice(["low", "mid", "high"], p=row) for row in p]
    return pd.DataFrame({"y": pd.Categorical(y), "x": x, "group": group})


CASES = [
    Case("gaussian_intercept", "y ~ 1", gaussian_data),
    Case("gaussian_numeric", "y ~ x", gaussian_data),
    Case("gaussian_two_numeric", "y ~ x + z", gaussian_data),
    Case("gaussian_categorical", "y ~ group", gaussian_data),
    Case("gaussian_numeric_categorical", "y ~ x + group", gaussian_data),
    Case("gaussian_interaction_numeric_numeric", "y ~ x * z", gaussian_data),
    Case("gaussian_interaction_numeric_categorical", "y ~ x * group", gaussian_data),
    Case("gaussian_cell_means", "y ~ 0 + group", gaussian_data),
    Case("gaussian_group_intercept", "y ~ x + (1|subject)", gaussian_data),
    Case("gaussian_group_slope", "y ~ x + (x|subject)", gaussian_data),
    Case("gaussian_group_interaction_numeric_numeric", "y ~ x * z + (x:z|subject)", gaussian_data),
    Case(
        "gaussian_group_interaction_numeric_categorical",
        "y ~ x * group + (x:group|subject)",
        gaussian_data,
    ),
    Case(
        "gaussian_group_interaction_categorical_categorical",
        "y ~ group * condition + (group:condition|subject)",
        gaussian_data,
    ),
    Case("gaussian_two_group_intercepts", "y ~ x + (1|subject) + (1|item)", gaussian_data),
    Case("gaussian_two_group_slopes", "y ~ x + (x|subject) + (z|item)", gaussian_data),
    Case(
        "gaussian_categorical_interaction",
        "y ~ group * condition",
        gaussian_data,
    ),
    Case(
        "gaussian_categorical_interaction_no_intercept",
        "y ~ 0 + group:condition",
        gaussian_data,
    ),
    Case("gaussian_spline", "y ~ bs(x, df=5)", spline_data),
    Case("gaussian_spline_plus_categorical", "y ~ bs(x, df=5) + group", spline_data),
    Case("gaussian_group_spline", "y ~ z + (bs(x, df=4)|subject)", spline_data),
    Case("bernoulli_numeric", "y ~ x", binary_data, family="bernoulli"),
    Case("bernoulli_categorical", "y ~ x + group", binary_data, family="bernoulli"),
    Case("poisson_numeric", "y ~ x", count_data, family="poisson"),
    Case("poisson_categorical", "y ~ x + group", count_data, family="poisson"),
    Case("negativebinomial_numeric", "y ~ x", count_data, family="negativebinomial"),
    Case("gamma_numeric", "y ~ x", gamma_data, family="gamma", kwargs={"link": "log"}),
    Case("gamma_categorical", "y ~ x + group", gamma_data, family="gamma", kwargs={"link": "log"}),
    Case("beta_numeric", "y ~ x", beta_data, family="beta"),
    Case("categorical_numeric", "y ~ x", categorical_response_data, family="categorical"),
    Case(
        "categorical_categorical", "y ~ x + group", categorical_response_data, family="categorical"
    ),
    Case(
        "gamma_distributional",
        bmb.Formula("y ~ x", "alpha ~ x"),
        gamma_data,
        family="gamma",
        kwargs={"link": {"mu": "log", "alpha": "log"}},
    ),
]


def select_cases(only: str | None, start: str | None) -> list[Case]:
    cases = CASES
    if start is not None:
        names = [case.name for case in cases]
        if start not in names:
            raise SystemExit(f"Unknown start case {start!r}. Options: {', '.join(names)}")
        cases = cases[names.index(start) :]
    if only is not None:
        requested = set(only.split(","))
        cases = [case for case in cases if case.name in requested]
        missing = requested - {case.name for case in cases}
        if missing:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")
    return cases


def summarize_idata(idata: az.InferenceData) -> str:
    parts = []
    for group in idata.groups():
        dataset = getattr(idata, group)
        variables = ", ".join(f"{name}{tuple(value.shape)}" for name, value in dataset.items())
        parts.append(f"{group}: {variables}")
    return "; ".join(parts)


def run_case(
    case: Case,
    prior_predictive: bool,
    prior_draws: int,
    posterior: bool,
    posterior_draws: int,
    posterior_tune: int,
) -> None:
    print(f"== {case.name}: {case.formula!s} [{case.family}]")
    model = case.make_model()
    model.build()
    print("   built")
    if prior_predictive:
        idata = model.prior_predictive(draws=prior_draws, random_seed=RNG_SEED)
        print(f"   prior predictive: {summarize_idata(idata)}")
    if posterior:
        with model.backend.model:
            idata = pm.sample(
                tune=posterior_tune,
                draws=posterior_draws,
                chains=1,
                cores=1,
                random_seed=RNG_SEED,
                progressbar=False,
            )
        print(f"   posterior: {summarize_idata(idata)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Comma-separated case names to run")
    parser.add_argument("--start", help="Run from this case onward")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--prior-predictive", action="store_true")
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--posterior", action="store_true")
    parser.add_argument("--posterior-draws", type=int, default=10)
    parser.add_argument("--posterior-tune", type=int, default=10)
    args = parser.parse_args()

    failures = []
    for case in select_cases(args.only, args.start):
        try:
            run_case(
                case,
                args.prior_predictive,
                args.draws,
                args.posterior,
                args.posterior_draws,
                args.posterior_tune,
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            failures.append((case.name, err))
            print(f"   FAILED: {type(err).__name__}: {err}")
            if not args.keep_going:
                raise

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"- {name}: {type(err).__name__}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
