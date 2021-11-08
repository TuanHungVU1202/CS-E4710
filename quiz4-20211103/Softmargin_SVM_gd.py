# thanks to https://github.com/tiepvupsu/ebookML_src/blob/master/src/softmargin_svm/softmargin%20SVM%20Example.ipynb
import numpy as np


class Softmargin_SVM_gd:
    # def loss(self, X, y, w, b):
    #     """
    #     X.shape = (2N, d), y.shape = (2N,), w.shape = (d,), b is a scalar
    #     """
    #     z = X.dot(w) + b  # shape (2N,)
    #     yz = y * z
    #     return (np.sum(np.maximum(0, 1 - yz)) + .5 * self.lam * w.dot(w)) / X.shape[0]

    def grad(self, X, y, w, lam):
        # z = X.dot(w) + b  # shape (2N,)
        z = X.dot(w)
        yz = y * z  # element wise product, shape (2N,)
        active_set = np.where(yz <= 1)[0]  # consider 1 - yz >= 0 only
        _yX = - X * y[:, np.newaxis]  # each row is y_n*x_n
        grad_w = (np.sum(_yX[active_set], axis=0) + lam * w) / X.shape[0]
        # grad_b = (-np.sum(y[active_set])) / X.shape[0]
        return grad_w

    def softmarginSVM_gd(self, X, y, w0, eta, no_iter, lam):
        w = w0
        # b = b0
        it = 0
        while it < no_iter:
            it = it + 1
            # (gw, gb) = self.grad(X, y, w, lam)
            gw= self.grad(X, y, w, lam)
            w -= eta * gw
            # b -= eta * gb
            # if (it % 1000) == 0:
            #     print('iter %d' % it + ' loss: %f' % self.loss(X, y, w, b))
        return w

    # w0 = .1 * np.random.randn(X.shape[1])
    # b0 = .1 * np.random.randn()
    # lr = 0.05
    # (w_hinge, b_hinge) = softmarginSVM_gd(X, y, w0, b0, lr)
    # print('w_hinge = ', w_dual)
    # print('b_hinge = ', b_dual)