#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with logistic regression in multilabel datasets

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
import logging
import numpy as np
from multiprocessing import Pool
import time

import pickle

from scalarization.multilabel import load_samples
from scalarization.mtl_scipy_scalar import ssSPMTLogisticRegression, \
                                           wsSPMTLogisticRegression,\
                                           nsSPMTLogisticRegression

from moopt import monise, nc, pgen, rennen, xnise, random_weights
import pygmo as pg

def moea(name, solsize, popsize, wscalar_, moea_type, max_gen=float('inf'), timeLimit=float('inf')):
    from platypus import Problem, TournamentSelector
    from platypus import NSGAII, NSGAIII, SPEA2
    
    from platyplus.operators import varOr, mutGauss, cxUniform
    from platyplus.types import RealGauss
    from platyplus.algorithms import SMSEMOA
    
    
    N = wscalar_.xdim
    M = wscalar_.M
    
    time_start = time.perf_counter()
    logger.info('Running '+moea_type+' in '+name)
    
    prMutation = 0.1
    prVariation = 1-prMutation
    
    vartor = varOr(cxUniform(), mutGauss(), prVariation, prMutation)
     
    def eval_(theta):
        return wscalar_.f(np.array(theta))
    
    problem = Problem(N, M)
    problem.types[:] = [RealGauss() for i in range(N)]
    problem.function = eval_
    
    if moea_type == 'NSGAII':
        alg = NSGAII(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor)
    elif moea_type == 'NSGAIII':
        alg = NSGAIII(problem, divisions_outer=3,
                      population_size=popsize,
                      selector=TournamentSelector(1),
                      variator=vartor)
    elif moea_type == 'SPEA2':
        alg = SPEA2(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor)
    elif moea_type == 'SMSdom':
        alg = SMSEMOA(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor,
                     selection_method = 'nbr_dom')
    elif moea_type == 'SMShv':
        alg = SMSEMOA(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor,
                     selection_method = 'hv_contr')
    gen = 1
    while gen<max_gen and time.perf_counter()-time_start<timeLimit:
        alg.step()
        gen+=1
    
    alg.population_size = solsize
    alg.step()

    moeaSols = [eval_(s.variables) for s in alg.result]

    moea_time = time.perf_counter() - time_start

    logger.info(moea_type+' in '+name+' finnished.')
    
    return moeaSols, moea_time


def runRandom(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = random_weights(weightedScalar=wscalar_, singleScalar=sscalar_,
                          targetSize=solsize)
    logger.info('Running WRAND in '+name)
    moo_.optimize()
    wrandSols = [sol.objs for sol in moo_.solutionsList]
    wrand_time = time.perf_counter() - time_start
    return wrandSols, wrand_time


def runMonise(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = monise(weightedScalar=wscalar_, singleScalar=sscalar_,
                  nodeTimeLimit=2,
                  targetSize=solsize, targetGap=0, nodeGap=0.01, norm=True)
    logger.info('Running MONISE in '+name)
    moo_.optimize()
    moniseSols = [sol.objs for sol in moo_.solutionsList]
    monise_time = time.perf_counter() - time_start

    return moniseSols, monise_time


def runPgen(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = pgen(weightedScalar=wscalar_, singleScalar=sscalar_,
                targetSize=solsize, norm=True, timeLimit=3600)
    logger.info('Running pgen in '+name)
    moo_.optimize()
    pgenSols = [sol.objs for sol in moo_.solutionsList]
    pgen_time = time.perf_counter() - time_start

    return pgenSols, pgen_time


def runXnise(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = xnise(weightedScalar=wscalar_, singleScalar=sscalar_,
                 targetSize=solsize, norm=True, timeLimit=3600)
    logger.info('Running pgen in '+name)
    moo_.optimize()
    xniseSols = [sol.objs for sol in moo_.solutionsList]
    xnise_time = time.perf_counter() - time_start

    return xniseSols, xnise_time


def runRennen(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = rennen(weightedScalar=wscalar_, singleScalar=sscalar_,
                  targetSize=solsize, norm=True, timeLimit=3600)
    logger.info('Running pgen in '+name)
    try:
        moo_.optimize()
        rennenSols = [sol.objs for sol in moo_.solutionsList]
        rennen_time = time.perf_counter() - time_start
    except SystemError:
        rennenSols = []
        rennen_time = -1

    return rennenSols, rennen_time

def runNoRennen(name, solsize, wscalar_, sscalar_):
    rennenSols = []
    rennen_time = -1

    return rennenSols, rennen_time


def runNC(name, solsize, nscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = nc(normalScalar=nscalar_, singleScalar=sscalar_,
              targetSize=solsize, norm='simple', timeLimit=3600)
    logger.info('Running NC in_'+name)
    moo_.optimize()
    ncSols = [sol.objs for sol in moo_.solutionsList]
    nc_time = time.perf_counter() - time_start

    return ncSols, nc_time

def runNCs(name, solsize, nscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = nc(normalScalar=nscalar_, singleScalar=sscalar_,
              targetSize=solsize, norm='sanchis', timeLimit=3600)
    logger.info('Running NC in_'+name)
    moo_.optimize()
    ncSols = [sol.objs for sol in moo_.solutionsList]
    nc_time = time.perf_counter() - time_start

    return ncSols, nc_time


ffunc = {'NSGAII': moea,
         'NSGAIII': moea,
         'SPEA2': moea,
         'SMS': moea,
         'SMSdom': moea,
         'random': runRandom,
         'monise': runMonise,
         'pgen': runPgen,
         'xnise': runXnise,
         'rennen': runRennen,
         'norennen': runNoRennen,
         'nc': runNC,
         'ncs': runNCs}


def safeRun(method, name, solsize, *args, rerun=False, **kwargs):
    full_name = method+'_'+name+'_'+str(solsize)
    try:
        if rerun:
            raise ValueError('rerun set to True.')
        with open('cache/'+full_name+'.pickle', 'rb') as f:
            sols_, time_ = pickle.load(f)

    except (FileNotFoundError, ValueError):
        sols_, time_ = ffunc[method](name, solsize, *args, **kwargs)
        with open('cache/'+full_name+'.pickle', 'wb') as f:
            pickle.dump((sols_, time_), f)

    return sols_, time_


def dom(sols, sol):
    for s in sols:
        if all(sol > s):
            return True
    return False


def ndomSols(sols_ref, sols):
    return [sol for sol in sols if not dom(sols_ref, sol)]


def nonNadir(nadir, sols):
    return [sol for sol in sols if all(sol <= nadir)]


def trainInstance(parameters):
    base, name, load_truetest, algorithms = parameters
    full_name = base+'_'+name
    DEBUG = False
    if DEBUG:
        targetTime = 300
        rerun=True
    else:
        targetTime = 3600
        rerun=False

    logger.info('Reading database '+full_name+'... ')
    Xaux, Yaux = load_samples.load_samples(name)
    M = Yaux.shape[1]

    X = []
    Y = []
    for label in range(Yaux.shape[1]):
        X += [Xaux]
        Y += [Yaux[:, [label]]]

    logger.info('Setting scalarizations ... ')
    solver_params = {'eps': 10**-4, 'max_iter': 10**3}
    sscalar_ = ssSPMTLogisticRegression(X, Y, solver_params=solver_params)
    wscalar_ = wsSPMTLogisticRegression(X, Y, solver_params=solver_params)
    nscalar_ = nsSPMTLogisticRegression(X, Y, solver_params=solver_params)


    solsize = 10*M
    
    results = {}
    
    for algorithm in algorithms:
        results[algorithm] = {}
        if algorithm in ['NSGAII', 'NSGAIII', 'SPEA2', 'SMS', 'SMSdom']:
            sols, time = safeRun(algorithm, name, solsize, solsize,
                                 wscalar_, algorithm, timeLimit=targetTime,
                                 rerun=rerun)
        elif algorithm in ['nc']:
            sols, time = safeRun(algorithm, name, solsize, nscalar_, sscalar_)
        else:
            sols, time = safeRun(algorithm, name, solsize, wscalar_, sscalar_)
        results[algorithm]['sols'] = sols
        results[algorithm]['time'] = time
        results[algorithm]['Nsols'] = len(sols)

    allSols = [np.array(sol) for algorithm in algorithms for sol in results[algorithm]['sols']]
    ndomS = ndomSols(allSols, allSols)

    nadir = np.max(ndomS, axis=0)
    utopia = np.min(ndomS, axis=0)
    nadir = nadir+0.01*abs(nadir)
    
    
    norm_nadir = (nadir - utopia)/(nadir - utopia)

    for algorithm in algorithms:
        logger.info('Calculating '+algorithm+' HV '+name)
        nonNadisSols = nonNadir(nadir, results[algorithm]['sols'])
        if len(nonNadisSols)!=0:
            while len(nonNadisSols)>solsize:
                hv = pg.hypervolume([(o-utopia)/(nadir-utopia) for o in nonNadisSols])
                best = np.argsort(hv.contributions(norm_nadir))[-solsize:]
                nonNadisSols = [o for i, o in enumerate(nonNadisSols) if i in best]                    
                
            hv = pg.hypervolume([(o-utopia)/(nadir-utopia) for o in nonNadisSols])
            results[algorithm]['HV'] = hv.compute(norm_nadir)
        else:
            results[algorithm]['HV'] = -1

    with open('results/convex_'+name+'.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+[algorithm for algorithm in algorithms])
        csv_.writerow(['HV']+[results[algorithm]['HV'] for algorithm in algorithms])
        csv_.writerow(['time']+[results[algorithm]['time'] for algorithm in algorithms])
        csv_.writerow(['Nsols']+[results[algorithm]['Nsols'] for algorithm in algorithms])
    

    return [[name]+[results[algorithm]['HV'] for algorithm in algorithms],
            [name]+[results[algorithm]['time'] for algorithm in algorithms],
            [name]+[results[algorithm]['Nsols'] for algorithm in algorithms]]

import csv


DEBUG = False

import signal, os

signal.signal(signal.SIGINT, lambda s, f : os.kill(os.getpid(), signal.SIGTERM))

def sig_handler(signum, frame):
    raise SystemError("Segfault")
    return None 

signal.signal(signal.SIGSEGV, sig_handler)
signal.signal(signal.SIGALRM, sig_handler)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    algorithms = [
                  'monise',
                  'random',
                  'nc',
                  'pgen',
                  'rennen',
                  'NSGAII',
                  'NSGAIII',
                  'SPEA2',
                  'SMSdom',
                  ]
    
    algorithms2 = [
                  'monise',
                  'random',
                  'nc',
                  'pgen',
                  'norennen',
                  'NSGAII',
                  'NSGAIII',
                  'SPEA2',
                  'SMSdom',
                  ]

    instances = ['emotions', 'flags' ,'yeast','birds','genbase']

    train_list = [('multilabel', name, True, algorithms)
                  for name in instances[:2]]+[('multilabel', name, True, algorithms2)
                  for name in instances[2:]]

    if DEBUG:
        outs = [trainInstance(l) for l in train_list]
    else:
        p = Pool(5)
        outs = p.map(trainInstance, train_list)

    outsHv = [out[0] for out in outs]
    outsTime = [out[1] for out in outs]
    outsSols = [out[2] for out in outs]

    with open('results/convex_HV.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsHv:
            csv_.writerow(row)

    with open('results/convex_time.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsTime:
            csv_.writerow(row)

    with open('results/convex_sols.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsSols:
            csv_.writerow(row)