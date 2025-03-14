from enum import Enum


class PlotType(Enum):
    REAL       = "real"
    IMAG       = "imag"
    ABS        = "abs"
    ANGLE      = "angle"
    REAL_IMAG  = "real_imag"
    UNCHANGED  = "unchanged"
    ABS_SQUARE = "abs_square"
