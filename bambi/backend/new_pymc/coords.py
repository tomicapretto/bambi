import numpy as np


def coords_from_response(term):
    # TODO: Build _levels
    # TODO: Let's try to not make this a 'family' thing, rather something the visitor decides.
    coords = {"__obs__": np.arange(term.shape[0])}
    if hasattr(term.family, "get_coords"):
        return term.family.get_coords(term)
    return {}


def coords_from_term(term):
    # Single numeric
    if term.kind == "numeric":
        if term.ndim == 1:
            return {}

        # A numeric that spans multiple columns (e.g., a spline)
        return {f"{term.name}_levels": np.arange(term.shape[1])}

    # Single categoric
    if term.kind == "categoric":
        if term.spans_intercept:
            name = f"{term.name}_levels"
        else:
            name = f"{term.name}_levels_reduced"
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

            # A categorical
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
    return coords_from_term(term.expr), coords_from_term(term.factor)


def coords_from_hsgp(term):
    # This handles univariate and multivariate cases
    coords = {f"{term.name}_weights_dim": np.arange(np.prod(term.m))}

    if term.by_levels is not None:
        coords[f"{term.name}_by"] = term.by_levels

    if not term.iso and term.shape[1] > 1:
        coords[f"{term.name}_var"] = np.arange(term.shape[1])

    return coords
