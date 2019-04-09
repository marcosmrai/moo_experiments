#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:57:39 2018

@author: marcos
"""

from moopt.scalarization_interface import scalar_interface, single_interface, w_interface, box_interface
from .scipy_scalar import weighted_scalar, bounded_weighted_scalar, single_scalar, normal_scalar, box_scalar
from .linear_models import SPMTLogisticRegression


class ssSPMTLogisticRegression(single_scalar, SPMTLogisticRegression,
                               single_interface, scalar_interface):
    pass


class wsSPMTLogisticRegression(weighted_scalar, SPMTLogisticRegression,
                               w_interface, scalar_interface):
    pass


class bwsSPMTLogisticRegression(bounded_weighted_scalar,
                                SPMTLogisticRegression, w_interface,
                                scalar_interface):
    pass


class nsSPMTLogisticRegression(normal_scalar, SPMTLogisticRegression,
                               scalar_interface):
    pass


class bsSPMTLogisticRegression(box_scalar, SPMTLogisticRegression,
                               box_interface, scalar_interface):
    pass

class NRSPMTLogisticRegression(SPMTLogisticRegression):
    @property
    def M(self):
        '''Property function'''
        return super().M-1

    def penalty_prox(self, w=None):
        return []


class ssNRSPMTLogisticRegression(single_scalar, NRSPMTLogisticRegression, single_interface, scalar_interface): pass
class wsNRSPMTLogisticRegression(weighted_scalar, NRSPMTLogisticRegression, w_interface, scalar_interface): pass
class bwsNRSPMTLogisticRegression(bounded_weighted_scalar, NRSPMTLogisticRegression, w_interface, scalar_interface): pass
class nsNRSPMTLogisticRegression(normal_scalar, NRSPMTLogisticRegression, scalar_interface): pass
class bsNRSPMTLogisticRegression(box_scalar, NRSPMTLogisticRegression, box_interface, scalar_interface): pass