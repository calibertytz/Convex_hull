import numpy as np
from scipy.optimize import minimize
from cvxopt import solvers, matrix
import gurobipy

def scipy_compute(target, convex_hull):
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

    return np.sqrt(res.fun)


def cvxopt_compute(target, convex_hull_base):
    target = np.array(target)
    convex_hull_base = np.array(convex_hull_base)

    l = convex_hull_base.shape[0]
    A = convex_hull_base - target
    P = matrix(np.dot(A, A.T), tc='d')
    q = matrix(np.zeros(l), tc='d')

    a = np.diag(-np.ones(l))
    b = np.ones((1, l))
    c = -np.ones((1, l))
    d = np.concatenate((a, b, c))

    G = matrix(d, tc='d')
    h = matrix(np.append(np.zeros(l), [1, -1]), tc='d')
    sol = solvers.qp(P, q, G, h)
    solvers.options['show_progress'] = False

    return np.sqrt(2 * sol['primal objective'])


def gurobi_compute(target, convex_base):
    A = convex_base - target
    var_num = convex_base.shape[0]

    B = A.dot(A.T)

    # create model
    MODEL = gurobipy.Model()
    MODEL.setParam('OutPutFlag', 0)
    MODEL.setParam("NonConvex", 2)

    # create vars
    x = MODEL.addVars(range(0, var_num), vtype=gurobipy.GRB.CONTINUOUS, name='x')

    # update envs
    MODEL.update()

    # set objective functions
    MODEL.setObjective(quicksum(quicksum(x[i] * B[i, j] for i in range(var_num)) * x[j] for j in range(var_num)),
                       sense=gurobipy.GRB.MINIMIZE)

    # add constraint
    MODEL.addConstr(quicksum(x) == 1)
    for i in range(var_num):
        MODEL.addConstr(x[i] >= 0)

    # optimize
    MODEL.optimize()

    return np.sqrt(MODEL.objVal)




if __name__ == '__main__':
    X = np.array([0, 0])
    Y = np.array([[0, 1], [1, 0]]) # result should be about 0.7

    print(f'scipy result: {scipy_compute(X, Y)} \n '
          f'cvxopt result: {cvxopt_compute(X, Y)} \n'
          f'gurobi result: {gurobi_compute(X, Y)}')
