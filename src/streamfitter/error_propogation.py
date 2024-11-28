from enum import StrEnum, auto


class ErrorPropogation(StrEnum):
    PROPOGATION = auto()
    JACKNIFE = auto()
    BOOTSTRAP = auto()
