# solve.py

# Imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import fmin_cg, approx_fprime
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

# fit_shard
def fit_shard(I, J, alpha, X, y, k, epsilon=1e-6, update_colours=False,
              limit_colours=True, check_gradients=False, **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    R0 = (I - J)
    y = np.copy(y)

    def f(x):
        shard = Shard(x.reshape(X.shape), k)
        H = shard(domain)
        R = R0 + alpha * (J - y) * H[..., np.newaxis]
        r = R.ravel()
        return 0.5 * np.dot(r, r)

    def fprime(x, return_energy=False):
        shard = Shard(x.reshape(X.shape), k)
        H, dX = shard(domain, return_dX=True, epsilon=epsilon)
        d = alpha * (J - y)
        R = R0 + d * H[..., np.newaxis]
        J_ = dX[..., np.newaxis] * d
        J_ = J_.reshape(X.size, -1).transpose()
        r = R.ravel()
        dx = np.dot(r, J_)

        if not return_energy:
            return dx
        else:
            return np.dot(r, r), dx

    if check_gradients:
        x = X.ravel()
        approx_D = approx_fprime(x, f, epsilon=1e-6)
        D = fprime(x)
        print 'approx_D: (%4g, %4g)' % (np.amin(approx_D), np.amax(approx_D))
        print 'D: (%4g, %4g)' % (np.amin(D), np.amax(D))
        atol = 1e-4
        print 'allclose (%g)?' % atol, np.allclose(approx_D, D, atol=atol)

    callbacks = []
    def callback_handler(xk):
        for callback in callbacks:
            callback(xk)

    states = []
    def save_state(xk):
        states.append((np.copy(xk).reshape(X.shape), 
                       np.copy(y)))
    callbacks.append(save_state)

    if update_colours:
        def update_colour(xk):
            y[:] = colour_shard(I, J, alpha, xk.reshape(X.shape), k, 
                                limit_colours=limit_colours)
        callbacks.append(update_colour)

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
