from typing import Sequence, Dict, List

import ast
import textwrap

from functools import singledispatch

import formulae as fm
import numpy as np

from bambi.transformations import HSGP


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
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
    """Wraps long strings into multiple lines.

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
    """Extract the names of the arguments passed to a function.

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
        If all criteria are met, the names of the arguments. Otherwise it returns None.
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


def clean_formula_lhs(x):
    """Remove the left hand side of a model formula and the tilde.

    Parameters
    ----------
    x : str
        A model formula that has '~' in it.

    Returns
    -------
    str
        The right hand side of the model formula
    """
    assert "~" in x
    position = x.find("~")
    return x[position + 1 :]


def get_auxiliary_parameters(family):
    """Get names of auxiliary parameters

    Obtains the difference between all the parameters and the parent parameter of a family.
    These parameters are known as auxiliary or nuisance parameters.

    Parameters
    ----------
    family : bambi.families.Family
        The family

    Returns
    -------
    set
        Names of auxiliary parameters in the family
    """
    return set(family.likelihood.params) - {family.likelihood.parent}


def get_aliased_name(term):
    """Get the aliased name of a model term

    Model terms have a name and, optionally, an alias. The alias is used as the "name" if it's
    available. This is a helper that returns the right "name".

    Parameters
    ----------
    term : BaseTerm
        The term

    Returns
    -------
    str
        The aliased name
    """
    if term.alias:
        return term.alias
    return term.name


def is_single_component(term) -> bool:
    """Determines if formulae term contains a single component"""
    return hasattr(term, "components") and len(term.components) == 1


def is_call_component(component) -> bool:
    """Determines if formulae component is the result of a function call"""
    return isinstance(component, fm.terms.call.Call)


def is_stateful_transform(component):
    """Determines if formulae call component is a stateful transformation"""
    return component.call.stateful_transform is not None


def is_hsgp_term(term):
    """Determines if formulae term represents a HSGP term

    Bambi uses this function to detect HSGP terms and treat them in a different way.
    """
    if not is_single_component(term):
        return False
    component = term.components[0]
    if not is_call_component(component):
        return False
    if not is_stateful_transform(component):
        return False
    return isinstance(component.call.stateful_transform, HSGP)


def remove_common_intercept(dm: fm.matrices.DesignMatrices) -> fm.matrices.DesignMatrices:
    """Removes the intercept from the common design matrix

    This is used in ordinal families, where the intercept is requested but not used because its
    inclusion, together with the cutpoints, would create a non-identifiability problem.
    """
    dm.common.terms.pop("Intercept")
    intercept_slice = dm.common.slices.pop("Intercept")
    dm.common.design_matrix = np.delete(dm.common.design_matrix, intercept_slice, axis=1)
    return dm


@singledispatch
def accept(node):
    return ast.unparse(node)


@accept.register(ast.BinOp)
def _(node):
    # TODO: how should this work for "-" symbols?
    # TODO: It does not work if there's no "Add" op
    if isinstance(node.op, ast.Add):
        return [accept(node.left), accept(node.right)]
    return ast.unparse(node)


def flatten_list(x):
    output = []
    for element in x:
        if isinstance(element, list):
            output += flatten_list(element)
        else:
            output.append(element)
    return output


def split_top_level_terms(formula):
    terms_list = accept(ast.parse(formula).body[0].value)
    return flatten_list(terms_list)


def get_terminal_names(expr) -> List[str]:
    """Get names of variables from an expression omitting names of callables

    Examples
    --------
    "a * f(x, y)" -> ['a', 'x', 'y']
    "a * f(x, f(y))" -> ['a', 'x', 'y'] 
    "a * f(x, f(y), np.exp(z)) + np.log(f(y), f(m))" -> ['a', 'x', 'y', 'm', 'y', 'z']
    """
    names = []
    ast_expr = ast.parse(expr)
    for node in ast.walk(ast_expr):
        if isinstance(node, ast.Name):
            names.append(node.id)

    for node in ast.walk(ast_expr):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.remove(node.func.value.id)
            elif isinstance(node.func, ast.Name):
                names.remove(node.func.id)
    return names


class NonLinearExpression:
    """NonLinearExpression for a Bambi model

    Parameters
    ----------
    expression : str
        A mathematical expression containing a non-linear equation.
    parameters : Sequence[str]
        The names of the parameters involved in the non-linear equation.
    variables : Sequence[str]
        The names of the data variables involved in the non-linear equation.
    formulas : Dict[str, str]
        A mapping between each parameter and a model formula indicating how each parameter is
        related to predictor variables.
    """
    def __init__(self, expression, parameters, variables, formulas: Dict[str, str]):
        if not set(formulas).issubset(set(parameters)):
            raise ValueError("At least one formula doesn't match any parameter.")
        
        self.expression = expression
        self.parameters = parameters
        self.variables = variables
        
        self.constant_parameters = []
        self.distributional_parameters = []

        for parameter in self.parameters:
            if parameter not in formulas:
                formulas[parameter] = f"{parameter} ~ 1"
                self.constant_parameters.add(parameter)
            else:
                self.distributional_parameters.add(parameter)
        self.formulas = formulas

    @property
    def callable(self):
        arguments = ", ".join(self.parameters + self.variables)
        code = f"lambda {arguments}: {self.expression}"
        return eval(code)

    def __repr__(self):
        parameters = ["    * " + param for param in self.parameters]
        variables = ["    * " + param for param in self.variables]
        body = (
            "\n"
            "  Expression\n" +
            "    " +  self.expression + "\n" +
            "  Parameters\n" + 
            "\n".join(parameters) + "\n"
            "  Variables\n" +
            "\n".join(variables)
        )
        return f"{self.__class__.__name__}{body}\n"
    

def extract_nlexprs(formula, nlpars=None) -> str:
    formula_out = ""
    nlexprs = []
    if nlpars is not None:
        expressions = split_top_level_terms(formula)
        for expression in expressions:
            names = get_terminal_names(expression)
            parameters = [name for name in names if name in nlpars]
            if not parameters:
                if formula_out == "":
                    formula_out += expression
                else:
                    formula_out += " + " + expression
                continue
            variables = [name for name in names if name not in parameters]
            nlexprs.append(NonLinearExpression(expression, parameters, variables, {}))

    return formula_out, nlexprs


class NonLinearParameter:
    def __init__(self, name, component, prefix):
        self._name = name
        self.component = component
        self.prefix = prefix

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self._name}"
        return self._name
    
    @property
    def alias(self):
        return self._alias

    @alias.setter
    def alias(self, value):
        assert isinstance(value, str), "Alias must be a string"
        self._alias = value
