from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from bambi.backend.pymc.coords import (
    coords_from_common,
    coords_from_group_specific,
    coords_from_hsgp,
)
from bambi.backend.pymc.types import Coords, Dims
from bambi.parameters import ConditionalParameter
from bambi.terms import CommonTerm, GroupSpecificTerm, HSGPTerm


@dataclass(frozen=True)
class CommonTermInfo:
    term: CommonTerm
    coords: Coords

    @property
    def data_dims(self) -> Dims:
        return ("__obs__", *self.coords)


@dataclass(frozen=True)
class GroupSpecificTermInfo:
    term: GroupSpecificTerm
    expression_coords: Coords
    factor_coords: Coords


@dataclass(frozen=True)
class HSGPTermInfo:
    term: HSGPTerm
    coords: Coords


@dataclass(frozen=True)
class GroupSpecificFactorInfo:
    factor_name: str
    factor_ndim: int
    terms: tuple[GroupSpecificTermInfo, ...]
    groups_n: int


@dataclass(frozen=True)
class ConditionalParameterInfo:
    parameter: ConditionalParameter
    common_terms: tuple[CommonTermInfo, ...]
    hsgp_terms: tuple[HSGPTermInfo, ...]
    group_specific_factors: tuple[GroupSpecificFactorInfo, ...]

    @property
    def label(self) -> str:
        return self.parameter.label

    @property
    def group_specific_terms(self) -> tuple[GroupSpecificTermInfo, ...]:
        return tuple(term for factor in self.group_specific_factors for term in factor.terms)


@dataclass(frozen=True)
class GroupSpecificFactorPlan:
    parameter_label: str
    factor_name: str
    factor_ndim: int
    terms: tuple[GroupSpecificTermInfo, ...]
    groups_index: np.ndarray
    groups_new: tuple[object, ...]
    groups_n: int


@dataclass
class GroupSpecificTermGraph:
    lookup: pt.Variable

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "GroupSpecificTermGraph":
        return GroupSpecificTermGraph(lookup=memo[self.lookup])


@dataclass
class GroupSpecificParameterGraph:
    contribution: pt.Variable
    contribution_dims: Dims
    terms: dict[str, GroupSpecificTermGraph] = field(default_factory=dict)

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "GroupSpecificParameterGraph":
        return GroupSpecificParameterGraph(
            contribution=memo[self.contribution],
            contribution_dims=self.contribution_dims,
            terms={label: term.clone(memo) for label, term in self.terms.items()},
        )


@dataclass
class GroupSpecificGraphState:
    parameters: dict[str, GroupSpecificParameterGraph] = field(default_factory=dict)

    def clone(self, memo: dict[pt.Variable, pt.Variable]) -> "GroupSpecificGraphState":
        return GroupSpecificGraphState(
            parameters={
                label: parameter.clone(memo) for label, parameter in self.parameters.items()
            }
        )


def make_conditional_parameter_info(parameter: ConditionalParameter) -> ConditionalParameterInfo:
    common_terms = tuple(
        CommonTermInfo(term=term, coords=coords_from_common(term))
        for term in parameter.common_terms.values()
    )
    hsgp_terms = tuple(
        HSGPTermInfo(term=term, coords=coords_from_hsgp(term))
        for term in parameter.hsgp_terms.values()
    )

    terms_by_factor = {}
    for term in parameter.group_specific_terms.values():
        expression_coords, factor_coords = coords_from_group_specific(term)
        term_info = GroupSpecificTermInfo(
            term=term,
            expression_coords=expression_coords,
            factor_coords=factor_coords,
        )
        terms_by_factor.setdefault(term.factor, []).append(term_info)

    group_specific_factors = tuple(
        GroupSpecificFactorInfo(
            factor_name=terms[0].term.factor_name,
            factor_ndim=len(terms[0].factor_coords),
            terms=tuple(terms),
            groups_n=len(terms[0].term.groups),
        )
        for terms in terms_by_factor.values()
    )
    return ConditionalParameterInfo(
        parameter=parameter,
        common_terms=common_terms,
        hsgp_terms=hsgp_terms,
        group_specific_factors=group_specific_factors,
    )
