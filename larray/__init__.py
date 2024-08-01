__version__ = '0.34.6'


from larray.core.axis import Axis, AxisCollection, X
from larray.core.group import Group, LGroup, LSet, IGroup, union
from larray.core.array import (Array, zeros, zeros_like, ones, ones_like, empty, empty_like, full,
                               full_like, sequence, labels_array, ndtest, asarray, identity, diag,
                               eye, all, any, sum, prod, cumsum, cumprod, min, max, mean, ptp, var,
                               std, median, percentile, stack, zip_array_values, zip_array_items)
from larray.core.session import Session, local_arrays, global_arrays, arrays
from larray.core.checked import CheckedArray, CheckedSession, CheckedParameters
from larray.core.constants import nan, inf, pi, e, euler_gamma
from larray.core.metadata import Metadata
from larray.core.ufuncs import wrap_elementwise_array_func, maximum, minimum, where
from larray.core.npufuncs import (sin, cos, tan, arcsin, arccos, arctan, hypot, arctan2, degrees,
                                  radians, unwrap, sinh, cosh, tanh, arcsinh, arccosh, arctanh,
                                  angle, real, imag, conj,
                                  round, around, rint, fix, floor, ceil, trunc,
                                  exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2,
                                  i0, sinc, signbit, copysign, frexp, ldexp,
                                  convolve, clip, sqrt, absolute, fabs, sign, fmax, fmin, nan_to_num,
                                  real_if_close, interp, isnan, isinf, inverse)
from larray.core.misc import isscalar

from larray.inout.misc import from_lists, from_string
from larray.inout.pandas import from_frame, from_series
from larray.inout.csv import read_csv, read_tsv, read_eurostat
from larray.inout.excel import read_excel
from larray.inout.hdf import read_hdf
from larray.inout.sas import read_sas
from larray.inout.stata import read_stata
from larray.inout.xw_excel import open_excel, Workbook
from larray.inout.xw_reporting import ExcelReport, ReportSheet

# module is imported only to initialize handlers for .pkl and .pickle (we del _pkl at the end of this module)
import larray.inout.pickle as _pkl

from larray.util.options import get_options, set_options

from larray.viewer import view, edit, debug, compare, run_editor_on_exception

from larray.extra.ipfp import ipfp

from larray.example import get_example_filepath, load_example_data, EXAMPLE_EXCEL_TEMPLATES_DIR

# this makes larray.random functions accessible as la.random.* without explicitly importing larray.random
import larray.random as _rnd

__all__ = [
    # axis
    'Axis', 'AxisCollection', 'X',
    # group
    'Group', 'LGroup', 'LSet', 'IGroup', 'union',
    # array
    'Array', 'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like', 'full',
    'full_like', 'sequence', 'labels_array', 'ndtest', 'asarray', 'identity', 'diag', 'eye',
    'all', 'any', 'sum', 'prod', 'cumsum', 'cumprod', 'min', 'max', 'mean', 'ptp', 'var', 'std',
    'median', 'percentile', 'stack', 'zip_array_values', 'zip_array_items',
    # session
    'Session', 'local_arrays', 'global_arrays', 'arrays',
    # constrained
    'CheckedArray', 'CheckedSession', 'CheckedParameters',
    # constants
    'nan', 'inf', 'pi', 'e', 'euler_gamma',
    # metadata
    'Metadata',
    # ufuncs
    'wrap_elementwise_array_func',
    'maximum', 'minimum', 'where',
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'hypot', 'arctan2', 'degrees', 'radians',
    'unwrap', 'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    'angle', 'real', 'imag', 'conj',
    'round', 'around', 'rint', 'fix', 'floor', 'ceil', 'trunc',
    'exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p', 'logaddexp', 'logaddexp2',
    'i0', 'sinc', 'signbit', 'copysign', 'frexp', 'ldexp',
    'convolve', 'clip', 'sqrt', 'absolute', 'fabs', 'sign', 'fmax', 'fmin', 'nan_to_num',
    'real_if_close', 'interp', 'isnan', 'isinf', 'inverse',
    # core/misc
    'isscalar',
    # inout
    'from_lists', 'from_string', 'from_frame', 'from_series', 'read_csv', 'read_tsv',
    'read_eurostat', 'read_excel', 'read_hdf', 'read_sas', 'read_stata',
    'open_excel', 'Workbook', 'ExcelReport', 'ReportSheet',
    # utils
    'get_options', 'set_options',
    # viewer
    'view', 'edit', 'debug', 'compare', 'run_editor_on_exception',
    # ipfp
    'ipfp',
    # example
    'get_example_filepath', 'load_example_data', 'EXAMPLE_EXCEL_TEMPLATES_DIR',
]


# ==== DEPRECATED API ====

# F401: imported item not used
from larray.core.axis import x                                                              # noqa: F401
from larray.core.group import PGroup                                                        # noqa: F401
from larray.core.array import (LArray, aslarray, create_sequential, ndrange, larray_equal,  # noqa: F401
                               larray_nan_equal, nan_equal, element_equal)
from larray.core.npufuncs import round_                                                     # noqa: F401

_deprecated_names = [
    # axis
    'x',
    # group
    'PGroup',
    # array
    'LArray', 'aslarray',
    'create_sequential', 'ndrange',
    'larray_equal', 'larray_nan_equal', 'nan_equal', 'element_equal',
    'round_',
]

__all__ += _deprecated_names

# cleanup namespace (module was imported only to initialize handlers for .pkl and .pickle)
del _pkl
del _rnd
