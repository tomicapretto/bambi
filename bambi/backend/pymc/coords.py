import numpy as np


def _label(term):
    return getattr(term, "label", term.name)


def _shape(term):
    return getattr(term, "shape", np.shape(term.data))


def _ndim(term):
    return getattr(term, "ndim", len(_shape(term)))


def coords_from_response(term, family):
    coords_data = {"__obs__": np.arange(term.shape[0])}
    coords = {}
    coords_reduced = {}

    if hasattr(family, "get_levels"):
        levels = family.get_levels(term)
    else:
        levels = term.levels

    if levels:
        coords[f"{term.label}_levels"] = levels
        if term.reference:
            # There's a restriction applied when there is a reference level
            levels_restricted = [level for level in levels if level != term.reference]
        else:
            levels_restricted = levels[1:]

        coords_reduced[f"{term.label}_levels_reduced"] = levels_restricted

    elif term.ndim > 1:
        # A multidimensional numeric outcome.
        # Both non-reduced and reduced dimensions are called the same.
        # Dict union will make sure we attempt to add the dimension only once.
        # We still need regular and reduced coords because that is what things such as
        # additive predictors expect.
        coords[f"{term.label}_levels"] = np.arange(term.shape[1])
        coords_reduced[f"{term.label}_levels"] = np.arange(term.shape[1])

    return coords_data, coords, coords_reduced


def coords_from_term(term):
    # Single numeric
    if term.kind == "numeric":
        if _ndim(term) == 1:
            return {}

        # A numeric that spans multiple columns (e.g., a spline)
        return {f"{_label(term)}_levels": np.arange(_shape(term)[1])}

    # Single categoric
    if term.kind == "categoric":
        if term.spans_intercept:
            name = f"{_label(term)}_levels"
        else:
            name = f"{_label(term)}_levels_reduced"
        return {name: term.levels}

    # Interaction
    if term.kind == "interaction":
        # All numerics, return empty dict
        if all(el.kind == "numeric" for el in term.components):
            return {}

        coords = {}
        for el in term.components:
            # A numeric that spans multiple columns (e.g., a spline)
            if el.kind == "numeric" and el.value.ndim == 2 and el.value.shape[1] > 1:
                coords[f"{el.name}_levels"] = np.arange(el.value.shape[1])

            if el.kind == "categoric":
                if el.spans_intercept:
                    name = f"{el.name}_levels"
                else:
                    name = f"{el.name}_levels_reduced"

                coords[name] = el.contrast_matrix.labels
        return coords

    # Otherwise
    return {}


def coords_from_common(term):
    return coords_from_term(term)


def coords_from_group_specific(term):
    group_specific_term = getattr(term, "term", term)
    return coords_from_term(group_specific_term.expr), coords_from_term(group_specific_term.factor)


def coords_from_hsgp(term):
    # This handles univariate and multivariate cases
    coords = {f"{term.label}_weights_dim": np.arange(np.prod(term.m))}

    if term.by_levels is not None:
        coords[f"{term.label}_by"] = term.by_levels

    if not term.iso and term.shape[1] > 1:
        coords[f"{term.label}_var"] = np.arange(term.shape[1])

    return coords
