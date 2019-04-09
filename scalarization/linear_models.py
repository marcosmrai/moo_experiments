"""
Machine learning linear losses
"""
"""
Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause

import numpy as np
import copy

LOGOFINF = 700  # exp(LOGOFINF)+eps <INF
EPS = 10**-323


def clean_beta(beta, shape, offset):
    assert shape is not None, 'shape should be informed.'
    if offset is None:
        return beta.reshape(shape)
    else:
        return beta.flat[offset:offset+np.prod(shape)].reshape(shape)


def fill_beta(vec, size, offset, beta=None):
    if size is None or offset is None:
        return vec
    else:
        if beta is None:
            beta = np.zeros((size, 1))
        else:
            beta = beta.reshape((size, 1))
        beta[offset: offset + vec.size, 0] = vec.flat[:]
        return beta


def calc(func):
    def wrapper(self, beta, **kwargs):
        beta = clean_beta(beta, self.shape, self.offset)
        return func(self, beta, **kwargs)
    return wrapper


def operation(func):
    def wrapper(self, beta, **kwargs):
        beta_ = clean_beta(beta, self.shape, self.offset)
        vec = func(self, beta_, **kwargs)
        return fill_beta(vec, beta.size, self.offset)
    return wrapper


def exp_sat(x):
    '''Saturated exponential - avoid inf returns'''
    return np.exp(np.minimum(x, LOGOFINF))


def log_sat(x):
    '''Saturated exponential - avoid inf returns'''
    return np.log(x+EPS)


def sigmoid(x):
    """Calculates de sigmoid function

    Parameters
    ----------
    x : ndarray, (N,1) or (N,)

    Returns
    -------
    sig : float
        The sigmoid
    """
    assert (len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[1] == 1),
            'x is not a vector')
    return 1 / (1 + exp_sat(-x))


def softmax(x):
    """Calculates de softmax function

    Parameters
    ----------
    x : ndarray, (N,K)

    Returns
    -------
    soft : ndarray, (N,K)
        The sigmoid
    """
    assert len(x.shape) == 2 and x.shape[1] >= 2, 'x is not a Nxd matrix, K>=2'
    exp_ = exp_sat(x)
    den = exp_.sum(axis=1).reshape(exp_.shape[0], 1)+EPS
    return exp_/den


class Prob():
    def __init__(self, X, Y, bias=True):
        self.__X = X
        self.__K = Y.shape[1]
        self._bias = bias

    @property
    def X(self):
        '''Property function'''
        return self.__X

    @property
    def K(self):
        '''Property function'''
        return self.__K

    def prob(self, theta, X=None):
        return self._prob(theta)
        '''Avoid calculations by matching hashes
           if inherited classes will not have hashes
           making this method behave like the implemented _prob'''
        return self._prob(theta)

    def param2D(self, param):
        '''Assert in the right dimension for the parameter'''
        N, d = self.X.shape
        if self._bias:
            assert param.size == (d+1)*self.K, 'reshape error'
            return param.reshape(d+1, self.K)
        else:
            assert param.size == d*self.K, 'reshape error'
            return param.reshape(d, self.K)


class LogisticProb(Prob):
    def _prob(self, theta, X=None):
        """Computes the probability of pertinence of any sample to any class.

        Parameters
        ----------
        theta : ndarray, shape (d,1) or (d+1,1)
            Coefficient vector.

        Returns
        -------
        pr : ndarray, shape (N,1)
            Estimated class probabilities.
        """
        if X is None:
            X = self.X

        N, d = X.shape
        theta = self.param2D(theta)

        if self._bias:
            pr = sigmoid(theta[0, :] + X @ theta[1:, :])
        else:
            pr = sigmoid(X@theta)
        return pr


class LogisticLoss(LogisticProb):
    def __init__(self, X, Y, sample_weights=None, class_weights=None,
                 bias=True, mean=True, offset=0, shape=None):
        N, d = X.shape
        self.__X = X

        assert type(Y) == np.ndarray, 'Y should be a ndarray'

        if Y.ndim == 1:
            self.__Y = Y.reshape(N, 1)
        else:
            self.__Y = Y

        assert (Y.ndim == 2 and Y.shape[0] == N and Y.shape[1] == 1,
                'Y should have only one task')
        self.__K = 1

        if sample_weights is None:
            self.__sw = np.ones((N, 1))
        else:
            assert sample_weights.size == N, 'wrong sw format'
            self.__sw = self.__sample_weights = sample_weights.reshape(N, 1)

        if class_weights is None:
            self.__W = self.__sw
        else:
            assert class_weights.size == 2, 'wrong cw format'
            self.__W = (Y * self.__sw * class_weights[0] +
                        (1 - Y) * self.__sw * class_weights[1])

        self._bias = True if bias is None else bias
        self.__avg = True if mean is None else mean

        self.shape = shape
        self.offset = offset

        self.reset()

    def reset(self):
        """Free any cached computations from previous use of this Function.
        From the interface "Function".
        """
        self._L = None

    @property
    def X(self):
        '''Property function'''
        return self.__X

    @property
    def Y(self):
        '''Property function'''
        return self.__Y

    @property
    def W(self):
        '''Property function'''
        return self.__W

    @property
    def Wsum(self):
        '''Property function'''
        return np.sum(self.__W, axis=1, keepdims=True)

    @property
    def K(self):
        '''Property function'''
        return self.__K

    @calc
    def f(self, theta):
        """Computes the objective

        Parameters
        ----------
        theta : ndarray, shape (d,) or (d + 1)
            Coefficient vector.

        Returns
        -------
        loss : float
        """
        N, d = self.X.shape

        pr = self.prob(theta)
        log_l = -self.W*(self.Y*log_sat(pr)+(1-self.Y)*log_sat(1-pr))

        div = N if self.__avg else 1
        losses = log_l.sum(0)/div
        return losses.sum()

    @operation
    def grad(self, theta):
        """Computes the gradient

        Parameters
        ----------
        theta : ndarray, shape (d,) or (d + 1)
            Coefficient vector.

        Returns
        -------
        grad : d(+1),1
            Coefficient grad.
        """
        N, d = self.X.shape

        pr = self.prob(theta)
        grad_aux = self.W*(pr - self.Y)

        div = N if self.__avg else 1
        if self._bias:
            grad_loss = np.append(np.ones((N, 1)).T@grad_aux,
                                  self.X.T@grad_aux,
                                  axis=0)/div
        else:
            grad_loss = self.X.T@grad_aux/div

        assert grad_loss.size == theta.size, 'grad error'
        return grad_loss.reshape(theta.size, 1)


class L2Squared():
    """The proximal operator of the squared L2 function with a penalty
    formulation

        f(\beta) = l * (0.5 * ||\beta||²_2 - c),

    where ||\beta||²_2 is the squared L2 loss function. The constrained
    version has the form

        0.5 * ||\beta||²_2 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            0.5 * ||\beta||²_2 <= c. The default value is c=0, i.e. the
            default is a regularised formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0, offset=0, shape=None):

        self.l = max(0.0, float(l))
        self.c = float(c)
        self.penalty_start = max(0, int(penalty_start))

        self.shape = shape
        self.offset = offset

    @calc
    def f(self, beta):
        """Function value.

        From the interface "Function".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (0.5 * np.dot(beta_.T, beta_)[0, 0] - self.c)

    @operation
    def grad(self, beta):
        """Gradient of the function.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
            grad = np.vstack((np.zeros((self.penalty_start, beta_.shape[1])),
                              self.l * beta_))
        else:
            beta_ = beta
            grad = self.l * beta_
        return grad


class SPMTLogisticRegression():
    def __init__(self, X=None, Y=None,
                 loss_params={},
                 penalized=False, penalty_params={},
                 solver='FISTA', solver_params={}):

        self.__loss_params = {'sample_weights': None, 'class_weights': None,
                              'bias': True, 'mean': True}
        self.__loss_params.update(loss_params)

        self.__penalized = penalized
        penalty_start = 1 if self.__loss_params['bias'] else 0
        self.__penalty_params = {'l': 1.0, 'c': 0.0,
                                 'penalty_start': penalty_start}
        self.__penalty_params.update(penalty_params)

        self.__solver = solver
        self.__solver_params = {'eps': 10**-4, 'max_iter': 10**2}
        self.__solver_params.update(solver_params)

        if X is not None and Y is not None:
            self.__X = X
            self.__Y = Y
            self.setup()

    @property
    def M(self):
        '''Property function'''
        return self.__M

    @property
    def T(self):
        '''Property function'''
        return self.__T

    @property
    def fobjs(self):
        return self.losses()+self.penalty_prox()

    def losses(self):
        losses = []
        for loss_params, X, Y in zip(self.task_loss_params, self.X, self.Y):
            losses += [LogisticLoss(X, Y, **loss_params)]
        return losses

    def penalty_prox(self):
        if self.__penalized:
            return [L2Squared(**self.penalty_params)]
        else:
            return []

    @property
    def X(self): return self.__X

    @property
    def Y(self): return self.__Y

    @property
    def classes(self): return self.__data.classes

    @property
    def binary(self): return self.__data.binary

    @property
    def binaryzed(self): return self.__data.binaryzed

    @property
    def solver_params(self): return self.__solver_params

    @property
    def penalty_params(self): return self.__penalty_params

    @property
    def penalty_shape(self): return self.__penalty_params['shape']

    @penalty_shape.setter
    def penalty_shape(self, x): self.__penalty_params['shape'] = x

    @property
    def penalty_offset(self): return self.__penalty_params['offset']

    @penalty_offset.setter
    def penalty_offset(self, x): self.__penalty_params['offset'] = x

    @property
    def loss_shape(self): return self.__loss_params['shape']

    @loss_shape.setter
    def loss_shape(self, x): self.__loss_params['shape'] = x

    @property
    def loss_offset(self): return self.__loss_params['offset']

    @loss_offset.setter
    def loss_offset(self, x): self.__loss_params['offset'] = x

    @property
    def solver(self): return self.__solver

    @property
    def penalty(self): return self.__penalty

    @property
    def loss_params(self): return self.__loss_params

    @property
    def class_weights(self): return self.__loss_params['class_weights']

    @class_weights.setter
    def class_weights(self, x): self.__loss_params['class_weights'] = x

    @property
    def theta(self): return self.__theta

    @property
    def x(self): return self.__theta.flat[:]

    @property
    def fit_runtime(self): return self.__fit_runtime

    @property
    def feasible(self): return self.__feasible

    @property
    def optimum(self): return self.__optimum

    @property
    def objs(self): return self.__objs

    @property
    def w(self): return self.__w

    @property
    def task_weights(self): return self.__task_weights

    @task_weights.setter
    def task_weights(self, x): self.__task_weights = x

    @property
    def xdim(self):
        '''Property function'''
        if self.loss_params['bias']:
            return self.Y[0].shape[1]*(self.X[0].shape[1]+1)
        else:
            return self.Y[0].shape[1]*self.X[0].shape[1]

    @property
    def prob_core(self):
        return self.__prob_core

    @property
    def fobjs(self):
        return self.losses()+self.penalty_prox()

    def f(self, theta=None):
        return np.array([Obj.f(theta) for Obj in self.fobjs])

    def theta_start(self, f, hotstart=[]):
        best_theta = np.zeros((self.xdim, 1))
        return best_theta

    def update(self, theta, objs, feasible, optimum, fit_runtime):
        self.__caching = False
        self.__theta = theta

        self.__objs = objs
        self.__feasible = feasible
        self.__optimum = optimum
        self.__fit_runtime = fit_runtime

    def setup(self):
        assert len(self.X) == len(self.Y), 'distinct number of taks on X and Y'
        assert (all([self.X[0].shape[1] == X.shape[1] for X in self.X]),
                'distinct number of attribuites in X')
        assert (all([self.Y[0].shape[1] == Y.shape[1] for Y in self.Y]),
                'distinct number of classes in Y')

        self.__T = len(self.X)
        self.__M = self.T + 1 if self.__penalized else self.T

        self.task_weights = np.ones(self.T)

        self.penalty_shape = (self.X[0].shape[1] +
                              int(self.loss_params['bias']),
                              self.Y[0].shape[1])
        self.loss_shape = (self.X[0].shape[1] + int(self.loss_params['bias']),
                           self.Y[0].shape[1])

        self.task_loss_params = []
        for tw, X, Y in zip(self.task_weights, self.X, self.Y):
            loss_params = copy.copy(self.loss_params)

            if loss_params['class_weights'] == np.ndarray:
                assert loss_params['class_weights'].ndim == 1 and \
                       loss_params['class_weights'].size == 2,\
                       'class_weights has the wrong format'
            elif loss_params['class_weights'] in ['default', None]:
                loss_params['class_weights'] = tw * np.ones(2)
            elif loss_params['class_weights'] in ['balanced']:
                N1 = np.maximum(1, Y.sum())
                N0 = np.maximum(1, (1 - Y).sum())
                loss_params['class_weights'] = (Y.shape[0] /
                                                (2 * np.array([N1, N0])))
            else:
                assert (loss_params['class_weights'].ndim == 1 and
                        loss_params['class_weights'] == Y.shape[1],
                        'class_weights has the wrong format')

            '''
            if loss_params['class_weights'] in ['default',None]:
                loss_params['class_weights']=tw

            else:
                assert type(loss_params['class_weights']) in [int, float],\
                'class_weights has the wrong format'

                loss_params['class_weights']= tw*loss_params['class_weights']
            '''

            self.task_loss_params += [loss_params]

        self.__prob_core = LogisticProb(self.X[0], self.Y[0],
                                        bias=self.loss_params['bias'])
