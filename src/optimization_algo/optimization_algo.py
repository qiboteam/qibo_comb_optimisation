from enum import Enum
from scipy.optimize import minimize

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


