from enum import Enum
from scipy.optimize import minimize
import numpy as np
from  ..optimization_class.quadratic_problem import quadratic_problem, linear_problem

class Result_status(Enum):  # I need to think further if we really need thi sclass.
    SUCCESS = 0
    FAILURE = 1
    INFEASIBLE = 2


'''
class OptimizationResult:

    def __init__(self, CircuitResult):
        self.sample = sample
        self.x =
        self.fx = # get the mode perhaps?
'''
class optimizer:

    def __init__(self, loss, initial_parameters, args, method, jac, hess, hessp, bounds, tol, callback, options):
        self.loss = loss
        self.initial_parameters = initial_parameters,
        self.arg = args
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hepps = hessp
        self.bounds, self.tol, self.callback, self.options = bounds, tol, callback, options


    def solve(self):
        m = minimize(
            self.loss,
            self.initial_parameters,
            self.args,
            self.method,
            self.jac,
            self.hess,
            self.hessp,
            self.bounds,
            self.constraints,
            self.tol,
            self.callback,
            self.options,
        )
        return m.x, m.fun, m

    def update_method(self, new_method):
        self.method = new_method

def two_ADMM_H(A0, A1, x, xbar, lambdak, rho, c, mu, eps, K_max, G, b, q):
    '''
    The optimizatin problem is f0(x) = q(x) + c/2||Gx-b||_2^2
    :param A0: numpy matrix forming part of constraint
    :param A1: numpy matrix forming part of constraint
    :param x0:
    :param y0:
    :param lambda0:
    :param rho:
    :param c:
    :param mu:
    :param eps:
    :param K_max:
    :return:
    '''
    k = 0
    linear_form = linear_problem(G, -b)
    tmp_quad = linear_form.square()
    tmp_quad = tmp_quad.multiply(c/2)
    fix_quadratic = q + tmp_quad

    while k < K_max and np.lnalg.norm(np.dot(A0, x) + np.dot(A1, xbar)) > eps:
        x =
        xbar =
        lambda_k = lambda_k + rho * (np.dot(A0,x) + np.dot(A1, xbar))



