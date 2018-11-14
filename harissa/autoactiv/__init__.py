"""
Autoactiv package
-----------------

Detection of positive loops in gene networks from single-cell data
"""
__version__ = '0.3'
__author__ = 'Ulysse Herbach'
__all__ = []

from .interface import posterior, infer, mapestim, get_param
__all__ += ['posterior', 'infer', 'mapestim', 'get_param']

from .scdata import scdata
__all__ += ['scdata']

from .model import model
__all__ += ['model']

from .graphics import plotHistoGenes
__all__ += ['plotHistoGenes']