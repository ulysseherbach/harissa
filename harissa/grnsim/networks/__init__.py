"""
Module for builing example networks for grnsim
"""
from .loader import hillnet, load
__all__ = ['hillnet', 'load']

from .cascade import cascade
__all__ += ['cascade']

from . import net0
network0 = load(net0, mode='bursty')
network0f = load(net0, mode='full')
__all__ += ['network0', 'network0f']