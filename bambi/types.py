from enum import StrEnum
from typing import NamedTuple


class Constraint(StrEnum):
    # NOTE: We may expand the restriction options
    REFERENCE = "reference"


class ResponseType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"


class CoefSpec(NamedTuple):
    ndim: int = 0
    constraint: Constraint | None = None


class ParamSpec(NamedTuple):
    # NOTE: Add transform? Such as ordered
    links: list[str]
    ndim: int = 0
    ordinal: bool = False
    coef_spec: CoefSpec = CoefSpec()


type Coords = dict[str, list[str]]
type Dims = tuple[str]
