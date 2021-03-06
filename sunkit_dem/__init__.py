# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

from .base_model import *
from .model_factory import *
