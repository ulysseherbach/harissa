"""
Autoactiv package
-----------------

Detection of positive loops in gene networks from single-cell data
"""
__version__ = '0.2'
__author__ = 'Ulysse Herbach'
__all__ = []

from .interface import posterior, mapestim, infer
__all__ += ['posterior', 'mapestim', 'infer']

from .scdata import scdata
__all__ += ['scdata']

from .model import model
__all__ += ['model']