# -*- coding: utf-8 -*-
'''Module that implements scalarizations using scipy
'''
import time

import numpy as np
import scipy.optimize as opt

from abc import ABCMeta, abstractmethod


class scipy_base(metaclass=ABCMeta):
    @abstractmethod
    def f(self): pass

    @abstractmethod
    def grad(self): pass


class general_scalar():
    def f(self, x, oSS=None):
        if oSS is None:
            return np.array([fobj.f(x) for fobj in self.fobjs])
        else:
            return np.array([fobj.f(x) for i, fobj in enumerate(self.fobjs)
                             if i in oSS])

    def grad(self, x, oSS=None):
        if oSS is None:
            return np.concatenate([fobj.grad(x) for fobj in self.fobjs],
                                  axis=1)
        else:
            return np.concatenate([fobj.grad(x)
                                   for i, fobj in enumerate(self.fobjs)
                                   if i in oSS],
                                  axis=1)

    def hessp(self, x, oSS=None):
        if oSS is None:
            return np.concatenate([fobj.hessp(x) for fobj in self.fobjs],
                                  axis=1)
        else:
            return np.concatenate([fobj.hessp(x)
                                   for i, fobj in enumerate(self.fobjs)
                                   if i in oSS],
                                  axis=1)


class weighted_scalar(general_scalar):
    '''Class that implements a weighted scalarization using scipy as solver'''
    @property
    def w(self):
        return self.__w

    def optimize(self, w, hotstart=[]):
        """Calculates the a multiobjective scalarization using weights

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
        """
        assert w.ndim == 1 and w.shape[0] == self.M, 'w format is wrong'
        start = time.clock()
        self.__w = w

        def f(x):
            return (w*self.f(x.reshape((x.size, 1)))).sum()

        def grad(x):
            return (w*self.grad(x.reshape((x.size, 1)))).sum(axis=1).flat[:]

        def fa(x):
            return (w*self.f(x.reshape((x.size, 1)))).sum()

        theta = self.theta_start(f, hotstart=hotstart).reshape((self.xdim, 1))

        opts = {
                # 'xtol': avextol,
                # 'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                # 'disp': True#,
                # 'iprint': 100
                # 'return_all': retall
                }

        out = opt.minimize(f, theta, jac=grad, method='L-BFGS-B',
                           tol=self.solver_params['eps'], options=opts)
        theta = out.x.reshape((self.xdim, 1))

        fit_runtime = time.clock() - start
        self.update(theta, self.f(theta), True,
                    True or np.linalg.norm(grad(theta)) <= 10^-10, fit_runtime)


class bounded_weighted_scalar(general_scalar):
    '''Class that implements a weighted scalarization using scipy as solver'''
    @property
    def w(self):
        return self.__w

    def optimize(self, w, u, hotstart=[]):
        """Calculates the a multiobjective scalarization using weights

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
        """
        assert w.ndim == 1 and w.shape[0] == self.M, 'w format is wrong'
        start = time.clock()
        self.__w = w

        def f(x):
            return (w*self.f(x.reshape((x.size, 1)))).sum()

        def grad(x):
            return (w*self.grad(x.reshape((x.size, 1)))).sum(axis=1).flat[:]

        def ineqF(j):
            return lambda x: -(self.f(x)[j]-u[j])

        def ineqG(j):
            return lambda x: -self.grad(x)[:, j]

        constraints = [{'type': 'ineq', 'fun': ineqF(j), 'jac': ineqG(j)}
                       for j in range(self.M)]

        theta = self.theta_start(f, hotstart=hotstart).reshape((self.xdim, 1))

        opts = {
                # 'xtol': avextol,
                # 'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                # 'disp': disp,
                # 'return_all': retall
                }

        out = opt.minimize(f, theta, jac=grad, constraints=constraints,
                           tol=self.solver_params['eps'], options=opts)

        theta = out.x.reshape((self.xdim, 1))

        fit_runtime = time.clock() - start
        self.update(theta, self.f(theta), True,
                    out.status == 0 or np.linalg.norm(grad(theta)) <= 10^-10,
                    fit_runtime)


class single_scalar(general_scalar):
    '''Class that implements a single objective scalarization using
    scipy as solver'''
    @property
    def w(self):
        return self.__w

    def optimize(self, i, hotstart=[]):
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
        assert type(i) == int and i < self.M, 'wrong i'
        self.__w = np.zeros(self.M)
        self.__w[i] = 1
        start = time.clock()

        def f(x):
            return float(self.f(x.reshape((x.size, 1)), [i]))

        def grad(x):
            return self.grad(x.reshape((x.size, 1)), [i]).flat[:]

        theta = self.theta_start(f, hotstart=hotstart).reshape((self.xdim, 1))

        opts = {
                # 'xtol': avextol,
                # 'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                # 'disp': disp,
                # 'return_all': retall
                }

        out = opt.minimize(f, theta, jac=grad, method='L-BFGS-B',
                           tol=self.solver_params['eps'], options=opts)

        theta = out.x.reshape((self.xdim, 1))
        fit_runtime = time.clock() - start
        self.update(theta, self.f(theta), True,
                    out.status == 0 or np.linalg.norm(grad(theta)) <= 10^-10,
                    fit_runtime)


class normal_scalar(general_scalar):
    def optimize(self, Xref, Ndir, l, T, solutionsList=None):
        start = time.clock()

        def nf(x, oSS=None):
            return (T@(self.f(x) - l)
                    if oSS is None else
                    T[oSS, :][:, oSS]@(self.f(x, oSS) - l[oSS]))

        def ng(x, oSS=None):
            return (self.grad(x)@T
                    if oSS is None else
                    self.grad(x, oSS)@T[oSS, :][:, oSS])

        def f_loss(x):
            return nf(x, [self.M-1])[-1]

        def f_grad(x):
            return ng(x, [self.M-1])[:, -1].flat[:]

        opts = {
                # 'xtol': avextol,
                # 'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                # 'disp': disp,
                # 'return_all': retall
                }

        def ineqF(j):
            return lambda x: -Ndir[:, j]@(nf(x)-Xref)

        def ineqG(j):
            return lambda x: -(ng(x)@Ndir[:, j]).flat[:]

        constraints = [{'type': 'ineq', 'fun': ineqF(j), 'jac': ineqG(j)}
                       for j in range(self.M-1)]

        x0 = np.ones(self.xdim)*10**-10
        out = opt.minimize(f_loss, x0, jac=f_grad, constraints=constraints,
                           tol=self.solver_params['eps'], options=opts)

        theta = out.x.reshape((self.xdim, 1))
        fit_runtime = time.clock() - start
        self.update(theta, self.f(theta), True,
                    (out.status == 0 or np.linalg.norm(f_grad(theta)) <= 10^-10),
                    fit_runtime)


class normal_scalarT(general_scalar):
    @property
    def x(self):
        return self.__x[:self.xdim]

    def mo_optimize(self, Xref, Ndir, l, T, X, Y, solutionsList=None):
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
        start = time.clock()
        oArgs = (X,Y)

        f = lambda x, oSS=None: self.f(x.reshape((x.size,1))) if oSS==None else self.f(x.reshape((x.size,1)), oSS)
        nf = lambda x, oSS=None: T@(f(x)-l) if oSS==None else T[oSS,:][:,oSS]@(f(x,oSS)-l[oSS])

        g = lambda x, oSS=None: self.grad(x.reshape((x.size,1))) if oSS==None else self.grad(x.reshape((x.size,1)), oSS)
        ng = lambda x, oSS=None: T@g(x) if oSS==None else T[oSS,:][:,oSS]@g(x,oSS)

        x = np.ones(self.xdim)*10**-10

        loss = lambda x: nf(x,[self.M-1])[-1]
        grad = lambda x: ng(x,[self.M-1])[:,-1].flat[:]

        #loss = lambda x: nf(x)[-1]
        #grad = lambda x: ng(x)[:,-1]

        opts = {#'xtol': avextol,
                #'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                #'disp': disp,
                #'return_all': retall
                }

        constraints = [{'type': 'ineq', 'fun': lambda x: -Ndir[:,j]@(nf(x)-Xref), 'jac': lambda x: -(ng(x)@Ndir[:,j]).flat[:]} for j in range(self.M-1)]

        out = opt.minimize(loss, x, jac=grad, constraints=constraints,
                               tol=self.solver_params['eps'], options=opts)

        self.__x = out.x.reshape((self.X.shape[1],self.Y.shape[1]))
        fit_runtime = time.clock() - start
        self.update(self.__x, self.f(self.__x), True,\
            out.status == 0 or np.linalg.norm(grad(self.__x)) <= 10^-10, fit_runtime)

class box_scalar(general_scalar):
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

    def __xParse(self, x):
        """Parse x to be used by baseOpt
        Parameters
        ----------
        x: array-like, shape (dim, )
            Argument used by the optimizer

        Returns
        -------
        x': array-like, shape (dim, )
            Argument used by baseOpt
        """
        return x[:self.xdim]

    @property
    def x(self):
        return self.__x[:self.xdim]


    def __bestIniGess(self, solutionsList):
        """Calcultes the best initial gess using a list of gesses - needs be implemented
        Parameters
        ----------
        x: array-like, shape (dim, )
            Argument used by the optimizer
        *oArgs: tuple
            Arguents used by baseOpt

        Returns
        -------
        feasIndex: float
            feasibility index
        """
        index = np.argmin([self.__calcAlpha(gess.x) for gess in solutionsList]+[self.__calcAlpha(self.x)])
        if index<len(solutionsList):
            gess = solutionsList[index]
            alpha=self.__calcAlpha(gess.x)
            self.__x[:-1]=gess.x.copy()
            self.__x[-1]=alpha
        #print('INDEX',index,' of ',len(solutionsList))


    def __getAlpha(self, x):
        """Calculates alpha
        Parameters
        ----------
        x: array-like, shape (dim, )
            Argument used by the optimizer

        Returns
        -------
        alpha: float
            Alpha is [0,1] and indicates how far the solution is from self.u
        """
        return x[self.xdim]

    def __calcAlpha(self, x):
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
        objs = self.f(self.__xParse(x))
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

    def __constr(self, x, i):
        """ Calculates if the constraint if violated <0
        Parameters
        ----------
        x: array-like, shape (dim, )
            Argument used by the optimizer
        i: int
            Index of the objective
        constrType: str
            Type of the constraint
        oArgs: tuple
            Arguents used by baseOpt
        Returns
        -------
        constr: float
            constraint violation
        """
        obj = self.f(self.__xParse(x), oSS = [i])
        alpha = x[self.xdim]
        return ((self.l + alpha*(self.u-self.l))[i]-obj)[0]


    def __gradConstr(self, x, i):
        """ Calculates the gradient of the constraint
        Parameters
        ----------
        x: array-like, shape (dim, )
            Argument used by the optimizer
        i: int
            Index of the objective
        constrType: str
            Type of the constraint
        oArgs: tuple
            Arguents used by baseOpt
        Returns
        -------
        grad: (dim, )
            gradient
        """
        grad = self.grad(self.__xParse(x), oSS = [i])
        return -np.append(grad,-(self.u[i]-self.l[i]))

    def __is_feasible(self):
        alpha = self.__calcAlpha(self.x)
        return alpha>=0 and alpha<=1

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
        start = time.clock()
        # unpacking arguments
        self.__l,self.__u=l, u

        opts = {#'xtol': avextol,
                #'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                #'disp': disp,
                #'return_all': retall
                }

        #building constraints
        constraints=[{'type': 'ineq', 'fun': self.__constr, 'jac': self.__gradConstr, 'args':(obj)}
            for obj in range(self.M)]

        # inicializing x
        self.__x = np.zeros(self.xdim+1)
        self.__x[self.xdim] = self.__calcAlpha(self.x)
        self.__bestIniGess(solutionsList)

        # building gradient
        fGrad = np.zeros(self.xdim+1)
        fGrad[-1] = 1
        alphagrad = lambda x: fGrad
        alphaobj = self.__getAlpha

        # optimizing
        out = opt.minimize(alphaobj, self.__x, jac=alphagrad, method='L-BFGS-B',
                           constraints=constraints, tol=self.solver_params['eps'], options=opts)

        # parsing output
        self.__x = out.x
        self.__x[self.xdim] = self.__calcAlpha(self.x)
        self.__alpha = self.__calcAlpha(self.x)
        self.__c = self.__calcCenter()


        loss = lambda x, *args: self.f(self.__xParse(x)).sum()
        grad = lambda x, *args: np.append(self.grad(self.__xParse(x)).sum(axis=1),[0])

        constraints+=[{'type': 'ineq', 'fun': lambda x: self.__alpha-alphaobj(x), 'jac': lambda x: -alphagrad(x)}]

        out = opt.minimize(loss, self.__x, jac=grad, method='L-BFGS-B',
                           constraints=constraints, tol=self.solver_params['eps'], options=opts)
        self.__x = out.x
        self.__x[self.xdim] = self.__calcAlpha(self.x)
        self.__alpha = self.__calcAlpha(self.x)
        self.__c = self.__calcCenter()

        theta = self.__xParse(self.__x)

        fit_runtime = time.clock() - start

        # updating father arguments
        self.update(theta, self.f(theta), True, self.__is_feasible(), fit_runtime)


class normal_scalar_old(general_scalar):
    def optimize(self, Xref, Ndir, l, T, solutionsList=None):

        start = time.clock()
        d = T.diagonal()

        f = lambda x, oSS=None: self.f(x.reshape((x.size,1))) if oSS==None else self.f(x.reshape((x.size,1)), oSS)
        nf = lambda x, oSS=None: d*(f(x)-l) if oSS==None else d[oSS]*(f(x,oSS)-l[oSS])

        g = lambda x, oSS=None: self.grad(x.reshape((x.size,1))) if oSS==None else self.grad(x.reshape((x.size,1)), oSS)
        ng = lambda x, oSS=None: d*g(x) if oSS==None else d[oSS]*g(x,oSS)

        x = np.ones(self.xdim)*10**-10

        loss = lambda x: nf(x,[self.M-1])[-1]
        grad = lambda x: ng(x,[self.M-1])[:,-1].flat[:]


        opts = {#'xtol': avextol,
                #'eps': self.eps,
                'maxiter': self.solver_params['max_iter'],
                #'disp': disp,
                #'return_all': retall
                }

        def ineqF(j):
            return lambda x: -Ndir[:,j]@(nf(x)-Xref)

        def ineqG(j):
            return lambda x: -(ng(x)@Ndir[:,j]).flat[:]

        constraints = [{'type': 'ineq', 'fun': ineqF(j), 'jac': ineqG(j)}  for j in range(self.M-1)]

        out = opt.minimize(loss, x, jac=grad, constraints=constraints,
                               tol=self.solver_params['eps'], options=opts)

        self.__x = out.x
        fit_runtime = time.clock() - start
        self.update(self.__x, self.f(self.__x), True,\
            out.status == 0 or np.linalg.norm(grad(self.__x)) <= 10^-10, fit_runtime)



    def fit(self, X=None, Y=None, w=None):
        raise('fit it is not suitable for scalarizations')