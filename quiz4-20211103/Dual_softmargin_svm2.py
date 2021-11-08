# thanks to https://github.com/SatyaVSarma/optimization-sdca-pegasos
import numpy as np
from numpy import asarray


class Dual_softmargin_svm2:
    # To compute the sum of individual hinge losses :
    def sum_hinge(self, y, X, w, n_samples):
        losses = np.fmax(np.zeros(n_samples), np.ones(n_samples) - (y * (X.dot(w))))
        return sum(losses)

    # To get the primal parameter w associated to a dual parameter alpha :
    def primal_param(self, X, alpha, lambd, n_samples):
        return (1 / (lambd * n_samples)) * (np.dot(np.transpose(X), alpha))

    # Used for computing the update of SDCA (see the report) :
    def get_delta_alpha_q(self, X, y, alpha, q, lambd, n_samples, w):
        A = (1 / (lambd * n_samples)) * (np.dot(np.transpose(X[q]), X[q]))
        B = np.dot(np.transpose(X[q]), w)
        delta_alpha_tilde_q = (y[q] - B) / A
        return y[q] * max(0, min(1, y[q] * (delta_alpha_tilde_q + alpha[q]))) - alpha[q]

    def SDCA_SVM(self, X, y, n_samples, T_0, lambd, nb_epochs=50, iteration="random", average=True,
                 keep_full_primal_history=True):

        # initialization of alpha to a vector of zeros
        alpha = np.zeros(n_samples)
        # initialization of histories
        w_history = []
        primal_history = []
        w = self.primal_param(X, alpha, lambd, n_samples)

        # main loop : random sampling
        if (iteration == "random"):
            for t in range(n_samples * nb_epochs):
                # pick random sample
                q = np.random.randint(0, n_samples)
                # compute and apply SDCA update rule
                delta_alpha_q = self.get_delta_alpha_q(X, y, alpha, q, lambd, n_samples, w)
                e = np.zeros(n_samples)
                e[q] = 1
                sdca_update = e * delta_alpha_q
                alpha = alpha + sdca_update
                w = self.primal_param(X, alpha, lambd, n_samples)

                # histories updates
                if (average == True):
                    w_history.append(w)
                if (keep_full_primal_history == True):
                    primal_history.append(
                        self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)
                # if it's too costly too keep the full primal history, keep at least some values to get an idea
                if (keep_full_primal_history == False and n_samples > 50000 and t % 50000 == 0):
                    primal_history.append(
                        self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)

        # main loop : permutation sampling
        elif (iteration == "permutation"):
            count = 0
            for t in range(nb_epochs):
                perm = np.random.permutation(n_samples)
                for q in perm:
                    # compute and apply SDCA update rule
                    delta_alpha_q = self.get_delta_alpha_q(X, y, alpha, q, lambd, n_samples, w)
                    e = np.zeros(n_samples)
                    e[q] = 1
                    sdca_update = e * delta_alpha_q
                    alpha = alpha + sdca_update
                    w = self.primal_param(X, alpha, lambd, n_samples)

                    # histories updates
                    if (average == True):
                        w_history.append(w)
                    if (keep_full_primal_history == True):
                        primal_history.append(
                            self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)
                    # if it's too costly too keep the full primal history, keep at least some values to get an idea
                    if (keep_full_primal_history == False and n_samples > 50000 and count % 50000 == 0):
                        primal_history.append(
                            self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)
                    count += 1

        # main loop : cyclic sampling
        elif (iteration == "cyclic"):
            count = 0
            # original
            # perm = np.random.permutation(n_samples)
            perm = n_samples
            for t in range(nb_epochs):
                # for q in perm:
                for q in range(n_samples):
                    # compute and apply SDCA update rule
                    delta_alpha_q = self.get_delta_alpha_q(X, y, alpha, q, lambd, n_samples, w)
                    e = np.zeros(n_samples)
                    e[q] = 1
                    sdca_update = e * delta_alpha_q
                    alpha = alpha + sdca_update
                    w = self.primal_param(X, alpha, lambd, n_samples)

                    # histories updates
                    if (average == True):
                        w_history.append(w)
                    if (keep_full_primal_history == True):
                        primal_history.append(
                            self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)
                    # if it's too costly too keep the full primal history, keep at least some values to get an idea
                    if (keep_full_primal_history == False and n_samples > 50000 and count % 50000 == 0):
                        primal_history.append(
                            self.sum_hinge(y, X, w, n_samples) / n_samples + (lambd / 2) * np.linalg.norm(w) ** 2)
                    count += 1

        if (average == True):
            return asarray(w_history[T_0:]).mean(axis=0), w_history, primal_history
        else:
            return w, primal_history