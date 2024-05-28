import cvxpy as cp
from scipy.linalg import sqrtm
import numpy as np

def goemans_williamson(edge):
    """
    :param edge: This take in an edge that describes the symmetric unweighted graph, starting from index 0
    :return: the optimal cut
    """
    n = 0
    for i, j in edge:
        n = max([n, i, j])
    n += 1
    X = cp.Variable((n,n), symmetric = True)
    constraints = [X>>0]
    constraints += [
        X[i,i] == 1 for i in range(n)
    ]
    objective = sum((1-X[i, j])/2 for (i,j) in edge)
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()
    x = sqrtm(X.value)
    u = np.random.randn(n) # this is a random normal vector
    x = np.sign(x@u)
    return x

#edges = [(0,1), (0,2), (1,3), (1,4), (2,3), (3,4)]
#print(goemans_williamson(edges))

