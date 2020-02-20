import warnings

import larray.extra.ipfp as ipfp

warnings.warn('ipfp function should be imported as "from larray import ipfp" or not imported explicitly at all if you '
              'use "from larray import *"', FutureWarning, stacklevel=2)
