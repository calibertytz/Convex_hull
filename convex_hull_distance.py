import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds


def compute(target, convex_hull):
    l = convex_hull.shape[0]

    def obj(x):
        result = target - sum(np.dot(np.diag(x), convex_hull))
        return np.linalg.norm(result)

    # 不等式约束
    ineq_cons = {"type": "ineq",
                 "fun": lambda x: x}

    # 等式约束
    eq_cons = {"type": "eq",
               "fun": lambda x: sum(x)-1}

    x0 = np.ones(l)/l

    res = minimize(obj, x0, method='SLSQP', constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': False})

    return res.fun




if __name__ == '__main__':
    #X, Y = np.random.random(size=(3, 4)), np.random.normal(size=4)
    Y = np.array([0, 0])
    X = np.array([[0, 1], [1, 0], [1, 1]])
    print(compute(target=Y, convex_hull=X))
