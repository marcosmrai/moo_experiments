#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xquared scalarization and performance experiment

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
import numpy as np

import logging
import scipy.optimize as opt

from moopt.scalarization_interface import scalar_interface, single_interface, w_interface
from moopt.monise import monise

class xsquared(w_interface, single_interface, scalar_interface):
    def __init__(self, d):
        self.__M = d.size
        self.__d = d

    @property
    def M(self):
        return self.__M

    @property
    def feasible(self):
        return True

    @property
    def optimum(self):
        return True

    @property
    def objs(self):
        return self.__objs

    @property
    def x(self):
        return self.__x

    @property
    def w(self):
        return self.__w

    def optimize(self, w, u=None):
        """Calculates the a multiobjective scalarization"""
        if type(w)==int:
            self.__w = np.zeros(self.M)
            self.__w[w] = 1
        elif type(w)==np.ndarray and w.ndim==1 and w.size==self.M:
            self.__w = w
        else:
            raise('w is in the wrong format')


        constraints = [{'type': 'eq', 'fun': lambda x: sum(x)-1, 'jac': lambda x: np.ones(self.M)}]

        def fg(i):
            m = np.zeros(self.M)
            m[i] = 1
            return lambda x: u[i]-(x[i]-self.__d[i])**2, lambda x: -m*(x-self.__d)

        if not u is None:
            for i in range(self.M):
                fc, gc = fg(i)
                constraints+=[{'type': 'ineq', 'fun': fc, 'jac': gc}]

        f = lambda x: self.__w@(x-self.__d)**2
        grad = lambda x: self.__w*(x-self.__d)

        out = opt.minimize(f, np.ones(self.M)/self.M, jac=grad, bounds=[(0., 1.)]*self.M,
                           constraints=constraints)

        self.__x = out.x
        self.__objs = (self.x-self.__d)**2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
import deap.tools._hypervolume.pyhv as hv

if __name__ == "__main__":
    M = 10
    d = np.random.rand(M)
    d = d/d.max()
    wscalar_ = sscalar_ = xsquared(d)
    node_gaps = [0.0001, 0.001, 0.01, 0.1]
    node_time_limits = [float('inf'), 600, 300, 120, 60, 30, 10, 5, 2, 1]

    hv_table = np.zeros((len(node_gaps), len(node_time_limits)))
    time_table = np.zeros((len(node_gaps), len(node_time_limits)))
    for j, node_time_limit in enumerate(node_time_limits):
        for i, node_gap in enumerate(node_gaps):
            time_start = time.perf_counter()
            moo_ = monise(weightedScalar = wscalar_, singleScalar = sscalar_,
                          targetGap=0.000, targetSize=5*M, redFact=float('inf'),
                          smoothCount=None, nodeTimeLimit=node_time_limit, nodeGap = node_gap)
            moo_.optimize()
            monise_time = time.perf_counter() - time_start

            moniseSols = [sol.objs for sol in moo_.solutionsList]
            nadir = np.max(moniseSols,axis=0)
            utopia = np.min(moniseSols,axis=0)
            moniseHV = hv.hypervolume([(o-utopia)/(nadir-utopia) for o in moniseSols],
                        (nadir-utopia)/(nadir-utopia))

            hv_table[i,j] = moniseHV
            time_table[i,j] = monise_time

    np.savetxt('results/analysis_HV.csv', hv_table)
    np.savetxt('results/analysis_time.csv', time_table)
