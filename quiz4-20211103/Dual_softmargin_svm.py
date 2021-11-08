# thanks to https://github.com/tiepvupsu/ebookML_src/blob/master/src/softmargin_svm/softmargin%20SVM%20Example.ipynb
import numpy as np
from cvxopt import matrix, solvers

class Dual_softmargin_svm:
    def get_w(self, X, y, lam, C):
        # build K
        V = np.concatenate((X0, -X1), axis = 0) # V[n,:] = y[n]*X[n]
        K = matrix(V.dot(V.T))
        p = matrix(-np.ones((2*N, 1)))
        # build A, b, G, h
        G = matrix(np.vstack((-np.eye(2*N), np.eye(2*N))))

        h = np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1))))
        h = matrix(np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1)))))
        A = matrix(y.reshape((-1, 2*N)))
        b = matrix(np.zeros((1, 1)))
        solvers.options['show_progress'] = False
        sol = solvers.qp(K, p, G, h, A, b)

        l = np.array(sol['x']).reshape(2*N) # lambda vector

        # support set
        S = np.where(l > 1e-5)[0]
        S2 = np.where(l < .999*C)[0]
        # margin set
        M = [val for val in S if val in S2] # intersection of two lists

        VS = V[S]           # shape (NS, d)
        lS = l[S]           # shape (NS,)
        w_dual = lS.dot(VS) # shape (d,)
        yM = y[M]           # shape(NM,)
        XM = X[M]           # shape(NM, d)
        b_dual = np.mean(yM - XM.dot(w_dual)) # shape (1,)
        print('w_dual = ', w_dual)
        print('b_dual = ', b_dual)