from typing import Collection, Sequence

import ast
import logging
import textwrap

import formulae as fm
import numpy as np
import pandas as pd
from xarray import DataTree

_log = logging.getLogger("bambi")


def listify(obj):
    """Wrap all non-list or tuple objects in a list

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def indentify(string: str, n: int = 2) -> str:
    """Add spaces to the beginning of each line in a multi-line string."""
    space = n * " "
    return space + space.join(string.splitlines(True))


def multilinify(sequence: Sequence[str], sep: str = ",") -> str:
    """Make a multi-line string out of a sequence of strings."""
    sep += "\n"
    return "\n" + sep.join(sequence)


def wrapify(string: str, width: int = 100, indentation: int = 2) -> str:
    """Wraps long strings into multiple lines

    This function is used to print the model summary.
    """
    lines = string.splitlines(True)
    wrapper = textwrap.TextWrapper(width=width)
    for idx, line in enumerate(lines):
        if len(line) > width:
            leading_spaces = len(line) - len(line.lstrip(" "))
            wrapper.subsequent_indent = " " * (leading_spaces + indentation)
            wrapped = wrapper.wrap(line)
            lines[idx] = "\n".join(wrapped) + "\n"
    return "".join(lines)


def extract_argument_names(expr, accepted_funcs):
    """Extract the names of the arguments passed to a function

    This is used to extract the labels from function calls such as `c(y1, y2, y3, y3)`.

    Parameters
    ----------
    expr : str
        An expression that is parsed to extract the components of the call.
    accepted_funcs : list
        A list with the names of the functions that we accept to parse.

    Returns
    -------
    list
        If all criteria are met, the names of the arguments. Otherwise it returns `None`.
    """
    # Extract the first thing in the body
    parsed_expr = ast.parse(expr).body[0]
    if not isinstance(parsed_expr, ast.Expr):
        return None

    # Check the value is a call
    value = parsed_expr.value
    if not isinstance(value, ast.Call):
        return None

    # Check call name is the name of an exepcted function
    if value.func.id not in accepted_funcs:
        return None

    # Check all arguments are either names or constants
    args = value.args
    if not all(isinstance(arg, ast.Name) for arg in args):
        return None

    # We can safely build labels now
    labels = [arg.id for arg in args]

    if labels:
        return labels
    return None


def split_formula(formula: str) -> tuple[str, str]:
    """Separate a formula into an intercept-only response formula and its predictor.

    Parameters
    ----------
    formula : str
        Formula in the form `response ~ predictor`.

    Returns
    -------
    response_formula : str
        Intercept-only formula used to build the response design.
    rhs : str
        Predictor from the right-hand side.

    Raises
    ------
    ValueError
        If the formula does not contain one response and one predictor.

    Examples
    --------
    >>> split_formula("y ~ x + z")
    ('y ~ 1', 'x + z')
    """
    lhs, separator, rhs = formula.partition("~")
    if not separator or not lhs.strip() or not rhs.strip() or "~" in rhs:
        raise ValueError("A formula must have the form 'response ~ predictor'.")
    return f"{lhs.strip()} ~ 1", rhs.strip()


def clean_formula_lhs(x):
    """Remove the left hand side of a model formula and the tilde

    Parameters
    ----------
    x : str
        A model formula that has '~' in it.

    Returns
    -------
    str
        The right hand side of the model formula.
    """
    assert "~" in x
    position = x.find("~")
    return x[position + 1 :]


def remove_common_intercept(dm: fm.matrices.DesignMatrices) -> fm.matrices.DesignMatrices:
    """Removes the intercept from the common design matrix

    This function is used with families for ordinal responses,
    where the intercept is requested but not used because its inclusion,
    together with the cutpoints, would create a non-identifiability problem.
    """
    dm.common.terms.pop("Intercept")
    intercept_slice = dm.common.slices.pop("Intercept")
    dm.common.design_matrix = np.delete(dm.common.design_matrix, intercept_slice, axis=1)
    return dm


def as_dataset(group):
    """Return a Dataset from either a DataTree group or a Dataset."""
    return group.to_dataset() if isinstance(group, DataTree) else group


def prepare_data(
    data: pd.DataFrame,
    formula,
    linear_formulas: Collection[str | None],
    categorical: str | list[str] | None = None,
    dropna: bool = False,
    parameters: Collection[str] = (),
    nonlinear_exprs=(),
) -> pd.DataFrame:
    """Convert categorical columns and align complete observations across model formulas.

    Parameters
    ----------
    data : pandas.DataFrame
        Original model data.
    formula : Formula
        Response and parameter formulas, with or without nonlinear expressions.
    linear_formulas : Collection of str or None
        Linear predictors already separated from nonlinear expressions.
    categorical : str or list of str, optional
        Columns to convert to categorical, in addition to all object columns.
    dropna : bool, optional
        Whether to remove rows with missing values in model inputs instead of raising an error.
    parameters : Collection of str, optional
        Names of likelihood parameters and nonlinear coefficients.
    nonlinear_exprs : Collection of NonlinearExpression, optional
        Already parsed nonlinear expressions.

    Returns
    -------
    pandas.DataFrame
        Prepared data. The supplied DataFrame is not modified.

    Raises
    ------
    ValueError
        If data is not a DataFrame, a required response is not specified, inputs contain missing
        values with `dropna=False`, or no complete observations remain.

    Examples
    --------
    Missing values in any parameter formula remove the same rows from the whole model:

    >>> from bambi import Formula
    >>> data = pd.DataFrame({"y": [1, 2, 3], "x": [0, 1, 2], "z": [0.1, None, 0.3]})
    >>> formula = Formula("y ~ x", "sigma ~ z")
    >>> prepared = prepare_data(data, formula, ("x", "z"), dropna=True)
    >>> prepared.index.tolist()
    [0, 2]
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas DataFrame.")

    # Convert object columns and requested categorical columns to categorical
    object_columns = list(data.select_dtypes("object").columns)
    to_convert = list(set(object_columns + listify(categorical)))
    if to_convert:
        data = data.copy()
        data[to_convert] = data[to_convert].apply(lambda x: x.astype("category"))

    parameter_names = set(parameters) | set(formula.nlpars)

    # Collect all data variables needed for the model
    data_vars = set()
    for nlexpr in nonlinear_exprs:
        data_vars.update(nlexpr.symbols - parameter_names)

    response_formula = formula.main
    if formula.is_nonlinear:
        response_formula, _ = split_formula(response_formula)

    response = fm.model_description(response_formula).response
    if response is None:
        raise ValueError(
            "No outcome variable is set! "
            "Please specify an outcome variable using the formula interface."
        )
    # `var_names` could have multiple names (i.e. counts(y1, y2, y3))
    data_vars.update(response.var_names)

    for linear_formula in linear_formulas:
        if linear_formula is None:
            # Skip formulas with only nonlinear expressions
            continue

        predictor_variables = fm.model_description(linear_formula).var_names
        if formula.is_nonlinear:
            predictor_variables = predictor_variables - parameter_names

        data_vars.update(predictor_variables)

    columns = list(data_vars & set(data.columns))
    incomplete = data[columns].isna().any(axis=1)
    if incomplete.any():
        if not dropna:
            raise ValueError(f"'data' contains {incomplete.sum()} incomplete rows.")
        _log.info(
            "Automatically removing %s/%s rows from the dataset.", incomplete.sum(), len(data)
        )
        data = data.loc[~incomplete].copy()

    if len(data) == 0:
        raise ValueError("'data' does not contain any complete observation.")

    return data


def prepare_new_data(
    model,
    data: pd.DataFrame,
    include_response: bool = False,
) -> pd.DataFrame:
    """Prepare new data for prediction or log likelihood evaluation.

    Parameters
    ----------
    model : Model
        Model whose parsed terms and expressions identify the required inputs.
    data : pandas.DataFrame
        New data. Missing grouping-factor values represent unknown groups.
    include_response : bool, optional
        Whether to check response inputs, as needed for log likelihood evaluation.

    Returns
    -------
    pandas.DataFrame
        Prepared data. The supplied DataFrame is not modified.

    Raises
    ------
    ValueError
        If data is not a DataFrame, required inputs contain missing values, or data is empty.

    Examples
    --------
    Missing grouping-factor values are allowed when predicting:

    >>> from bambi import Model
    >>> training = pd.DataFrame(
    ...     {"y": [1, 2, 1, 2], "x": [1, 2, 3, 4], "group": ["a", "a", "b", "b"]}
    ... )
    >>> model = Model("y ~ x + (1 | group)", training)
    >>> data = pd.DataFrame({"x": [1, 2], "group": ["a", None]})
    >>> prepared = prepare_new_data(model, data)
    >>> prepared.index.tolist()
    [0, 1]
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas DataFrame.")

    object_columns = list(data.select_dtypes("object").columns)
    categorical = model.data.select_dtypes("category").columns.intersection(data.columns)
    to_convert = list(set(object_columns) | set(categorical))
    if to_convert:
        data = data.copy()
        data[to_convert] = data[to_convert].apply(lambda x: x.astype("category"))

    data_vars = set()
    for parameter in model.parameter_graph.expression_nodes.values():
        data_vars.update(parameter.data_names)

    if include_response:
        data_vars.update(model.response_term.term.var_names)

    for term in model.get_terms().values():
        if hasattr(term, "factor"):
            data_vars.update(term.expr.var_names)
        else:
            data_vars.update(term.var_names)

    columns = list(data_vars & set(data.columns))
    incomplete = data[columns].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(f"'data' contains {incomplete.sum()} incomplete rows.")
    if len(data) == 0:
        raise ValueError("'data' does not contain any complete observation.")

    return data
