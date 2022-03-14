#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset generation and loading

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""

import numpy as np
import os
import yaml



data_g={}

FILE_NAME = os.path.basename(__file__)
DATA_PATH = os.path.abspath(__file__).replace(FILE_NAME, "")

def cache_folder():
    '''Returns the folder of the cache'''
    path = DATA_PATH+'bases/'
    if not os.access(path, os.F_OK):
        os.mkdir(path)
    return path


class data_ks():
    def __init__(self, name=None, M=2, N=100, cap_rate=0.5, 
                 obj_rel='CONFL', obj_scale='flat', weights_distr='uniform'):
        
        self.B = 1000
        
        if name is not None:
            self.name = name
            if self.load():
                return
            _, M, N, cap_rate, obj_rel, obj_scale, weights_distr = name.split('_')
            
            print(DATA_PATH+'bases/'+self.name+'.yaml')
            
            M = int(M[1:])
            N = int(N[1:])
            cap_rate = float(cap_rate[1:])
            print('Creating dataset with ',M,' objectives,',
                  N,' items, capacity ',cap_rate,' ',obj_rel,' ',obj_scale)
            ans = input('Confirm? Y/N: ')
            if ans == 'Y':
                print('Creating ...')
            else:
                print('Aborting ...')
                return
    
        self.M = M
        self.N = N
        self.cap_rate = cap_rate
        self.obj_rel = obj_rel
        self.obj_scale = obj_scale
        self.weights_distr = weights_distr
        
        self.name = ('ks_M'+str(self.M)+'_N'+str(self.N)+'_C'+str(self.cap_rate)+'_'
                     +self.obj_rel+'_' +self.obj_scale+'_'+self.weights_distr)
        
        if self.load():
            return

        if obj_scale == 'flat':
            self.s = np.linspace(1,1,self.M)
        if obj_scale == 'linear':
            self.s = np.linspace(0.01,1,self.M)
        if obj_scale == 'exp':
            self.s = 10**np.linspace(0,self.M-1,self.M)
        
        self.item_bonus = np.random.randint(self.M, size=self.N)
        self.item_bonus = self.M*np.ones(self.N)
        for k in range(self.M):
            self.item_bonus[np.random.randint(self.N, size=3)]=k
            
        self.name = ('ks_M'+str(self.M)+'_N'+str(self.N)+'_C'+str(self.cap_rate)+'_'
                     +self.obj_rel+'_' +self.obj_scale+'_'+self.weights_distr)
        
        self.gen()
        self.save()
        
    def gen(self):
        if self.weights_distr == 'uniform':
            self.weights = np.random.randint(self.B, size=self.N)
        self.capacity = int(self.cap_rate*self.B*self.N)//2
        self.values = {k:np.zeros(self.N) for k in range(self.M)}
        if self.obj_rel == 'RANDU':
            for k in range(self.M):
                self.values[k] = self.s[k]*np.random.randint(self.B, size=self.N)
        elif self.obj_rel == 'RANDE':
            for k in range(self.M):
                self.values[k] = self.s[k]*self.B*np.random.exponential(0.5, size=self.N)
        elif self.obj_rel == 'CONFL':
            for i in range(self.N):
                exps = np.random.permutation(self.M)+1
                for k in range(self.M):
                    self.values[k][i]=self.s[k]*self.B*np.random.exponential(exps[k])
            
        
        for k in range(self.M):
            self.values[k] = self.values[k].tolist()
        
            
        self.weights = self.weights.tolist()
            
    def save(self):
        instance = {}
        instance['M'] = self.M
        instance['N'] = self.N
        instance['capacity'] = self.capacity
        instance['weights'] = self.weights
        instance['values'] = self.values
    
        with open(DATA_PATH+'bases/'+self.name+'.yaml', 'w', encoding='utf8') as outfile:
            yaml.dump(instance, outfile, default_flow_style=False, allow_unicode=True)
            
    def load(self):
        if not os.path.exists(DATA_PATH+'bases/'+self.name+'.yaml'):
            return False
        
        with open(DATA_PATH+'bases/'+self.name+'.yaml', 'r') as stream:
            instance = yaml.safe_load(stream)
            
        self.M = instance['M']
        self.N = instance['N']
        self.capacity = instance['capacity']
        self.weights = [1 if w==0 else w for w in instance['weights']]
        self.values = instance['values']
        
        return True

def gen_many():
    for obj_rel in ['CONFL', 'RAND']:
        for M in [5, 10, 15, 20, 25]:
            M=M
            N=100
            cap_rate=0.5
            obj_rel=obj_rel
            obj_scale='flat' if obj_rel=='RAND' else 'random'
            weights_distr='exp'
            
            #name = ('ks_M'+str(M)+'_N'+str(N)+'_C'+str(cap_rate)+'_'
            #         +obj_rel+'_' +obj_scale+'_' +weights_distr)
            #data_ks(name)
            data_ks(M=M, N=N, cap_rate=cap_rate, obj_rel=obj_rel,
                    obj_scale=obj_scale if obj_rel=='RAND' else 'random', weights_distr=weights_distr)

if __name__=='__main__':
    gen_many()
