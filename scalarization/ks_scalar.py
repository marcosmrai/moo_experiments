# -*- coding: utf-8 -*-
'''Module that implements scalarizations using scipy
'''
import numpy as np
import gurobipy as grb

from moopt.scalarization_interface import scalar_interface, w_interface,\
                                          single_interface, box_interface


def opt_code(code_n):
    code_s = ['LOADED',
              'OPTIMAL',
              'INFEASIBLE',
              'INF_OR_UNBD',
              'UNBOUNDED',
              'CUTOFF',
              'ITERATION_LIMIT',
              'NODE_LIMIT',
              'TIME_LIMIT',
              'SOLUTION_LIMIT',
              'INTERRUPTED',
              'NUMERIC',
              'SUBOPTIMAL',
              'INPROGRESS']
    return code_s[code_n-1]


class ks_base():
    def __init__(self, instance):
        self.__instance = instance
        self.__M = instance.M
        self.N = instance.N
        self.xdim = self.N
        self.weights = instance.weights
        self.values = instance.values
        self.capacity = instance.capacity

    @property
    def M(self):
        return self.__M

    @property
    def optMessage(self):
        return self.__optMessage

    @property
    def feasible(self):
        return self.__feasible

    @property
    def optimum(self):
        return self.__optimal

    @property
    def objs(self):
        return self.__objs

    @property
    def x(self):
        return self.__x

    @property
    def seq(self):
        try:
            return self.__seq
        except:
            self.__seq = np.argsort([np.prod([self.values[i][j]
                                     for i in range(self.__M)]
                                             )/self.weights[j]
                                     for j in range(self.N)])[::-1]
            return self.__seq

    @property
    def max_perf(self):
        try:
            return self.__max_perf
        except:
            self.__max_perf = np.zeros(self.M)
            for k in range(self.M):
                payouts = [self.values[k][i]/self.weights[i] for i in range(self.N)]
                max_pay = np.argsort(payouts)[::-1]
                filled = 0
                for arg in max_pay:
                    if filled>=self.capacity:
                        break
                    self.__max_perf[k]+=np.ceil(min(self.weights[arg], self.capacity-filled)*payouts[arg])
                    filled+=self.weights[arg]
            return self.__max_perf

    def norm_fobj(self, x):
        fobj = np.array(self.fobj(x))
        return ((self.max_perf+fobj)/self.max_perf).tolist()

    def fobj(self, x):
        seq = self.seq
        sseq = [s for s in seq if x[s]]
        cuml = np.cumsum([self.weights[s] for s in sseq])
        piv = np.searchsorted(cuml, self.capacity, side='right')
        sseq = sseq[:piv]
        
        return [-sum([self.values[k][s] for s in sseq])
                for k in range(self.__M)]

    def start_model(self, M, N, weights, values, capacity):
        # Create a guroby model
        m = grb.Model("knapsack")

        # Creation of linear integer variables
        knap = {}

        for i in range(N):
            knap[i] = m.addVar(vtype=grb.GRB.BINARY, name='knap_'+str(i))
        m.update()

        # Inherent constraints of this problem

        # knapsack maximum load constraint
        expr = grb.quicksum(knap[i]*weights[i] for i in range(int(N)))
        cons = capacity
        m.addConstr(expr <= cons)

        m.update()

        # Objective functions
        F = [-grb.quicksum(knap[j]*values[i][j]
             for j in range(N))
             for i in range(M)]

        return m, knap, F

    def opt_model(self, m, knap, F):
        m.params.OutputFlag = False
        m.params.TimeLimit = 60
        m.params.MIPGapAbs = 0.01
        m.params.Threads = 1

        try:
            m.optimize()
            self.__optMessage = opt_code(m.status)
            self.__x = np.array([knap[i].x for i in range(self.N)])
            self.__objs = self.fobj(self.__x.tolist())
            ''' self.__objs = np.array([F[i].getValue()
            for i in range(self.__M)])'''
            self.__feasible = self.__optMessage in ['OPTIMAL', 'SUBOPTIMAL',
                                                    'LIMIT']
            self.__optimal = self.__optMessage == 'OPTIMAL'
        except:
            self.__optMessage = 'ERROR'
            self.__x = 'error'
            self.__objs = 'error'
            self.__feasible = False
            self.__optimal = False


class weighted_scalar(ks_base, w_interface, scalar_interface):
    '''Class that implements a weighted scalarization using scipy as solver'''
    def __init__(self, instance):
        super().__init__(instance)

    @property
    def w(self):
        '''Property method'''
        return self.__w

    @property
    def x(self):
        '''Property method'''
        return self.__x[:self.xdim]

    def optimize(self, w, solutionsList=None):

        if len(w) != self.M:
            raise ValueError('''Lenght of w must be equal to K\n got
                             (len(w),self.K): (%d %d)''' % (len(w), self.M))
        self.__w = w

        m, knap, F = self.start_model(self.M, self.N, self.weights, self.values,
                                      self.capacity)

        escF = grb.quicksum(F[i]*self.w[i] for i in range(self.M))
        m.setObjective(escF, grb.GRB.MINIMIZE)

        self.opt_model(m, knap, F)

        for i in range(self.M):
            m.addConstr(F[i] <= self.objs[i])

        escF = grb.quicksum(F[i] for i in range(self.M))
        m.setObjective(escF, grb.GRB.MINIMIZE)

        self.opt_model(m, knap, F)
        del m


class single_scalar(ks_base, single_interface, scalar_interface):
    '''Class that implements a single objective scalarization using
    scipy as solver'''
    def __init__(self, instance):
        super().__init__(instance)

    @property
    def w(self):
        return self.__w

    @property
    def x(self):
        return self.__x[:self.xdim]

    def optimize(self, i, solutionsList=None):
        self.__w = np.array([0]*self.M)
        self.__w[i] = 1

        m, knap, F = self.start_model(self.M, self.N, self.weights, 
                                      self.values, self.capacity)

        escF = F[i]
        m.setObjective(escF, grb.GRB.MINIMIZE)

        self.opt_model(m, knap, F)

        for i in range(self.M):
            m.addConstr(F[i] <= self.objs[i])

        escF = grb.quicksum(F[i] for i in range(self.M))
        m.setObjective(escF, grb.GRB.MINIMIZE)

        del m


class normal_scalar(ks_base, scalar_interface):
    def __init__(self, instance):
        super().__init__(instance)

    def optimize(self, Xref, Ndir, l, T, solutionsList=None):
        '''Calculates the a multiobjective scalarization using weights

        Parameters
        ----------
        w: array-like, shape (k,)
            Objetive weights

        oArgs: tuple
            Arguments used by baseOpt

        Returns
        -------
        self : object
            Returns self.
        '''
        m, knap, F = self.start_model(self.M, self.N, self.weights, 
                                      self.values, self.capacity)
        nF = [grb.quicksum(T[i, j]*(F[j]-l[j])
              for j in range(self.M))
              for i in range(self.M)]

        for j in range(self.M-1):
            expr = grb.quicksum(Ndir[i][j]*(nF[i]-Xref[i])
                                for i in range(self.M))
            m.addConstr(expr <= 0)

        m.update()

        escF = F[-1]
        m.setObjective(escF, grb.GRB.MINIMIZE)

        self.opt_model(m, knap, F)
        del m


class box_scalar(ks_base, box_interface, scalar_interface):
    def __init__(self, instance):
        super().__init__(instance)

    @property
    def u(self):
        return self.__u

    @property
    def l(self):
        return self.__l

    @property
    def c(self):
        return self.__c

    @property
    def alpha(self):
        return self.__alpha

    @property
    def x(self):
        return self.__x[:self.xdim]

    def __calcAlpha(self, objs):
        """Calculates alpha
        Parameters
        ----------
        x: array-like, shape (xdim, )
            Argument used by the optimizer
        oArgs: tuple
            Arguents used by baseOpt

        Returns
        -------
        alpha: float
            Alpha is [0,1] and indicates how far the solution is from self.u
        """
        return max((objs-self.l)/(self.u-self.l))

    def __calcCenter(self):
        """Calculates the center of the solution
        Parameters
        ----------

        Returns
        -------
        center: array-like, shape (dim, )

        """
        return self.l + self.__alpha*(self.u-self.l)

    def optimize(self, l, u, solutionsList=None):
        """Calculates the a multiobjective scalarization using boxes

        Parameters
        ----------
        l: array-like, shape (k,)
            lower bound of the box

        u: array-like, shape (k,)
            upper bound of the box

        globalL: array-like, shape (k,)
            global lower bound of the box

        globalU: array-like, shape (k,)
            global upper bound of the box

        oArgs: tuple
            Arguments used by baseOpt

        Returns
        -------
        self : object
            Returns self.
        """
        # unpacking arguments
        self.__l, self.__u = l, u

        m, knap, F = self.start_model(self.M, self.N, self.weights, 
                                      self.values, self.capacity)

        alpha = m.addVar(vtype=grb.GRB.CONTINUOUS, name='alpha')
        m.update()

        for i in range(self.M):
            expr = F[i]
            cons = self.l[i] + alpha*(self.u[i]-self.l[i])
            m.addConstr(expr <= cons)
        m.update()

        escF = alpha
        m.setObjective(escF, grb.GRB.MINIMIZE)

        self.opt_model(m, knap, F)
        self.__alpha = alpha.x
        self.__c = self.__calcCenter()
        del m
