#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with knapsack problem

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
import logging
import numpy as np
import time
from multiprocessing import Pool
import csv

import pickle

from scalarization.ks import ks_data
from scalarization.ks_scalar import weighted_scalar, single_scalar, \
                                    normal_scalar, box_scalar

from moopt import monise, nc, pgen, rennen, xnise, random_weights, esse
import pygmo as pg

def moea(name, solsize, popsize, wscalar_, moea_type, max_gen=float('inf'), timeLimit=float('inf')):
    from platypus import HUX, BitFlip, TournamentSelector
    from platypus import Problem, Binary
    from platypus import NSGAII, NSGAIII, SPEA2
    
    from platyplus.operators import varOr
    from platyplus.algorithms import SMSEMOA
    
    time_start = time.perf_counter()
    logger.info('Running '+moea_type+' in '+name)
    
    prMutation = 0.1
    prVariation = 1-prMutation
    
    vartor = varOr(HUX(), BitFlip(1), prVariation, prMutation)
    
    def evalKnapsack(x):
        return wscalar_.fobj([xi[0] for xi in x])
    
    problem = Problem(wscalar_.N, wscalar_.M)
    problem.types[:] = [Binary(1) for i in range(wscalar_.N)]
    problem.function = evalKnapsack
    
    
    if moea_type in ['NSGAII', 'NSGAII-2', 'NSGAII-4']:
        alg = NSGAII(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor)
    elif moea_type in ['NSGAIII', 'NSGAIII-2', 'NSGAIII-4']:
        alg = NSGAIII(problem, divisions_outer=3,
                      population_size=popsize,
                      selector=TournamentSelector(1),
                      variator=vartor)
    elif moea_type in ['SPEA2', 'SPEA2-2', 'SPEA2-4']:
        alg = SPEA2(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor)
    elif moea_type in ['SMSdom']:
        alg = SMSEMOA(problem, population_size=popsize,
                     selector=TournamentSelector(1),
                     variator=vartor,
                     selection_method = 'nbr_dom')
    elif moea_type in ['SMShv']:
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

    moeaSols = [evalKnapsack(s.variables) for s in alg.result]

    moea_time = time.perf_counter() - time_start

    logger.info(moea_type+' in '+name+' finnished.')
    
    return moeaSols, moea_time


def runRandom(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = random_weights(weightedScalar=wscalar_, singleScalar=sscalar_,
                          targetSize=solsize)
    logger.info('Running WRAND in '+name)
    moo_.optimize()
    wrandSols = [np.array(sol.objs) for sol in moo_.solutionsList]
    wrand_time = time.perf_counter() - time_start
    return wrandSols, wrand_time


def runESSE(name, solsize, box_scalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = esse(boxScalar=box_scalar_, singleScalar=sscalar_,
                          targetSize=solsize)
    logger.info('Running ESSE in '+name)
    moo_.optimize()
    esseSols = [np.array(sol.objs) for sol in moo_.solutionsList]
    esse_time = time.perf_counter() - time_start
    return esseSols, esse_time

def runMonise(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = monise(weightedScalar=wscalar_, singleScalar=sscalar_,
                  nodeTimeLimit=2,
                  targetSize=solsize, targetGap=0, nodeGap=0.01, norm=True)
    logger.info('Running MONISE in '+name)
    moo_.optimize()
    moniseSols = [np.array(sol.objs) for sol in moo_.solutionsList]
    monise_time = time.perf_counter() - time_start

    return moniseSols, monise_time


def runPgen(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = pgen(weightedScalar=wscalar_, singleScalar=sscalar_,
                targetSize=solsize, norm=True, timeLimit=3600)
    logger.info('Running PGEN in '+name)
    moo_.optimize()
    pgenSols = [np.array(sol.objs) for sol in moo_.solutionsList]
    pgen_time = time.perf_counter() - time_start

    return pgenSols, pgen_time


def runXnise(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = xnise(weightedScalar=wscalar_, singleScalar=sscalar_,
                 targetSize=solsize, norm=True)
    logger.info('Running xnise in '+name)
    moo_.optimize()
    xniseSols = [np.array(sol.objs) for sol in moo_.solutionsList]
    xnise_time = time.perf_counter() - time_start

    return xniseSols, xnise_time


def runRennen(name, solsize, wscalar_, sscalar_):
    time_start = time.perf_counter()
    moo_ = rennen(weightedScalar=wscalar_, singleScalar=sscalar_,
                  targetSize=solsize, norm=True, timeLimit=3600)
    logger.info('Running rennen in '+name)
    try:
        moo_.optimize()
        rennenSols = [np.array(sol.objs) for sol in moo_.solutionsList]
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


ffunc = {'NSGAII': moea,
         'NSGAIII': moea,
         'SPEA2': moea,
         'SMSdom': moea,
         'NSGAII-2': moea,
         'NSGAIII-2': moea,
         'SPEA2-2': moea,
         'SMShv': moea,
         'random': runRandom,
         'monise': runMonise,
         'pgen': runPgen,
         'xnise': runXnise,
         'rennen': runRennen,
         'norennen': runNoRennen,
         'nc': runNC,
         'esse':runESSE}


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


import signal

def sig_handler(signum, frame):
    raise SystemError("Segfault")
    return None 

signal.signal(signal.SIGSEGV, sig_handler)

def trainInstance(parameters):
    name, algorithms = parameters
    DEBUG = False
    if DEBUG:
        targetTime = 300
        rerun=True
    else:
        targetTime = 3600
        rerun=False

    logger.info('Reading database '+name+'... ')
    data_ = ks_data.data_ks(name)

    sscalar_ = single_scalar(data_)
    wscalar_ = weighted_scalar(data_)
    nscalar_ = normal_scalar(data_)
    box_scalar_ = box_scalar(data_)

    solsize = 10*wscalar_.M
    
    results = {}
    
    for algorithm in algorithms:
        results[algorithm] = {}
        if algorithm in ['NSGAII', 'NSGAIII', 'SPEA2', 'SMShv', 'SMSdom']:
            sols, time = safeRun(algorithm, name, solsize, solsize,
                                 wscalar_, algorithm, timeLimit=targetTime,
                                 rerun=rerun)
        elif algorithm in ['NSGAII-2', 'NSGAIII-2', 'SPEA2-2', 'SMS-2']:
            sols, time = safeRun(algorithm, name, solsize, 2*solsize,
                                 wscalar_, algorithm, timeLimit=targetTime,
                                 rerun=rerun)
        elif algorithm in ['nc']:
            sols, time = safeRun(algorithm, name, solsize, nscalar_, sscalar_)
        elif algorithm in ['esse']:
            sols, time = safeRun(algorithm, name, solsize, box_scalar_, sscalar_)
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

    with open('results/ks_'+name+'.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+[algorithm for algorithm in algorithms])
        csv_.writerow(['HV']+[results[algorithm]['HV'] for algorithm in algorithms])
        csv_.writerow(['time']+[results[algorithm]['time'] for algorithm in algorithms])
        csv_.writerow(['Nsols']+[results[algorithm]['Nsols'] for algorithm in algorithms])
    

    return [[name]+[results[algorithm]['HV'] for algorithm in algorithms],
            [name]+[results[algorithm]['time'] for algorithm in algorithms],
            [name]+[results[algorithm]['Nsols'] for algorithm in algorithms]]


DEBUG = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    algorithms = [
    		  'esse',
                  'monise',
                  'random',
                  #'nc',
                  #'pgen',
                  #'xnise',
                  #'rennen',
                  #'NSGAII',
                  #'NSGAIII',
                  #'SPEA2',
                  #'SMSdom',
                  ]
    
    algorithms2 = [
    		  'esse',
                  'monise',
                  'random',
                  #'nc',
                  #'pgen',
                  #'xnise',
                  #'norennen',
                  #'NSGAII',
                  #'NSGAIII',
                  #'SPEA2',
                  #'SMSdom',
                  ]
    
    #algorithms2 = algorithms = ['monise']

    instances = []
    for M in [5, 10, 15]:#, 20]:#, 25]:
        for obj_rel in ['RANDU']:#['EXP', 'CONFL', 'RAND'][:1]:
            for obj_scale in ['flat']:
                for cap_rate in [0.02, 0.04]:
                    M=M
                    N=100
                    obj_rel=obj_rel
                    weights_distr='uniform'
                    
                    ks_data.data_ks(M=M, N=N, cap_rate=cap_rate, obj_rel=obj_rel,
                            obj_scale=obj_scale, weights_distr=weights_distr)
                    
                    instances += ['ks_M'+str(M)+'_N'+str(N)+'_C'+str(cap_rate)+'_'
                                  +obj_rel+'_' +obj_scale+'_' +weights_distr]

    train_list = [(name, algorithms) for name in instances[:4]]+[(name, algorithms2) for name in instances[4:]]

    if DEBUG:
        outs = [trainInstance(l) for l in train_list]
    else:
        p = Pool(7)
        outs = p.map(trainInstance, train_list)

    outsHv = [out[0] for out in outs]
    outsTime = [out[1] for out in outs]
    outsSols = [out[2] for out in outs]

    with open('results/ks_HV_1000.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsHv:
            csv_.writerow(row)

    with open('results/ks_time_1000.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsTime:
            csv_.writerow(row)

    with open('results/ks_sols_1000.csv', 'w') as csvfile:
        csv_ = csv.writer(csvfile, delimiter=',', quotechar='"',
                          quoting=csv.QUOTE_MINIMAL)
        csv_.writerow(['']+algorithms)
        for row in outsSols:
            csv_.writerow(row)
