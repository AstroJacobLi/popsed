# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound
import warnings
warnings.simplefilter('ignore')


# Version
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


__name__ = 'popsed'
__author__ = ['Jiaxuan Li']
<<<<<<< HEAD
__all__ = ["utils", "mock", "speculator"]
=======
__all__ = ["utils", "mock"]
>>>>>>> d02aea9 (minor)
