# solve.py

# Imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fmin_cg
from shard import Shard

# subaxes
def subaxes(n):
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(float(n) / rows))
    w = 1.0 / cols
    h = 1.0 / rows

    f = plt.figure()
    axs = []
    for i in xrange(n):
        r = (rows - 1) - i / cols
        c = i % cols
        ax = f.add_axes([c * w, r * h, w, h], frameon=False)
        axs.append(ax)

    if len(axs) == 1:
        axs = axs[0]

    return f, axs

# shard_gradient
def shard_gradient(I, J, alpha, X, y, k, epsilon=1e-6):
    shape = I.shape[:2]
    domain = shape[::-1]
    shard = Shard(X, k)

    H, dX = shard(domain, return_dX=True, epsilon=epsilon)

    d = alpha * (J - y)
    R = (I - J) + d * H[..., np.newaxis]
    J = dX[..., np.newaxis] * d
    J = J.reshape(X.size, -1).transpose()
    dX = np.dot(R.ravel(), J)
    return dX.reshape(X.shape)

# fit_shard
def fit_shard(I, J, alpha, X, y, k, epsilon=1e-6, **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    d = alpha * (J - y)
    R0 = (I - J)

    def f(x):
        shard = Shard(x.reshape(X.shape), k)
        H = shard(domain)
        R = R0 + d * H[..., np.newaxis]
        r = R.ravel()
        return np.dot(r, r)

    def fprime(x, return_energy=False):
        shard = Shard(x.reshape(X.shape), k)
        H, dX = shard(domain, return_dX=True, epsilon=epsilon)
        R = R0 + d * H[..., np.newaxis]
        J = dX[..., np.newaxis] * d
        J = J.reshape(X.size, -1).transpose()
        r = R.ravel()
        dx = np.dot(r, J)

        if not return_energy:
            return dx
        else:
            return np.dot(r, r), dx

    callbacks = []
    def callback_handler(xk):
        for callback in callbacks:
            callback(xk)

    states = []
    def save_state(xk):
        states.append(np.copy(xk).reshape(X.shape))

    callbacks.append(save_state)

    xk = X.ravel()
    save_state(xk)

    kwargs['disp'] = 0
    try:
        callbacks.append(kwargs.pop('callback'))
    except KeyError:
        pass

    x = fmin_cg(f, xk, fprime, callback=callback_handler, **kwargs)

    return x.reshape(X.shape), states

# colour_shard
def colour_shard(I, J, alpha, X, k, limit_colours=True):
    shape = I.shape[:2]
    domain = shape[::-1]

    shard = Shard(X, k)
    H = shard(domain)

    B = alpha * H
    K = I - J * (1.0 - B[..., np.newaxis])

    b = B.ravel()
    bTb = np.dot(b, b)

    y = np.empty(K.shape[-1], dtype=np.float64)
    for i in xrange(K.shape[-1]):
        y[i] = np.dot(K[..., i].ravel(), b) / bTb

    if limit_colours:
        y[y > 1.0] = 1.0
        y[y < 0.0] = 0.0

    return y

# main_test_shard_gradient
def main_test_shard_gradient():
    INPUT_PATH = 'data/180,180-40,40,90,127,140,40-1,0,0-2.png'
    I = plt.imread(INPUT_PATH).astype(np.float64)
    shape = I.shape[:2]
    domain = shape[::-1]

    X = np.r_[40.0, 50, 100, 127, 140, 40].reshape(-1, 2)
    y = np.r_[1.0, 0.0, 0.0]
    k = 1.0
    alpha = 0.5

    # `J` is the initial image which the shard is blended with
    J = np.zeros_like(I)
    dX = shard_gradient(I, J, alpha, X, y, k)

    def apply_gradient(eta):
        X_ = X + eta * dX
        shard = Shard(X_, k)

        H = shard(domain)
        d = alpha * (J - y)
        R = (I - J) + d * H[..., np.newaxis]
        r = R.ravel()
        E = np.dot(r, r)
        print '%.7g -> %.7g' % (eta, E)
        return H

    etas = np.arange(20) * -0.05
    f, axs = subaxes(etas.shape[0])

    for i, eta in enumerate(etas):
        H = apply_gradient(eta)
        Ji = (J * (1.0 - alpha * H[..., np.newaxis]) 
              + alpha * y * H[..., np.newaxis])
        axs[i].imshow(Ji)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()

if __name__ == '__main__':
    main_test_shard_gradient()
    
