# solve.py

# Imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import approx_fprime, leastsq
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
def fit_shard(I, J, alpha, X, y, k, epsilon=1e-6, xtol=1e-4,
              check_gradients=False,
              **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    R0 = (I - J)

    def f(x):
        shard = Shard(x.reshape(X.shape), k)
        H = shard(domain)
        R = R0 + alpha * (J - y) * H[..., np.newaxis]
        return R.ravel()

    def Dfun(x):
        shard = Shard(x.reshape(X.shape), k)
        H, dX = shard(domain, return_dX=True, epsilon=epsilon)
        d = alpha * (J - y)
        R = R0 + d * H[..., np.newaxis]
        jac = dX[..., np.newaxis] * d
        return jac.reshape(X.size, -1).transpose()

    if check_gradients:
        x = X.ravel()
        def e(x):
            r = f(x)
            return 0.5 * np.dot(r, r)
        approx_D = approx_fprime(x, e, epsilon=epsilon)
        J_ = Dfun(x)
        r = f(x)
        D = np.dot(r, J_)
        print 'approx_D: (%4g, %4g)' % (np.amin(approx_D), np.amax(approx_D))
        print 'D: (%4g, %4g)' % (np.amin(D), np.amax(D))
        atol = 1e-4
        print 'allclose (atol=%g)?' % atol, np.allclose(approx_D, D, atol=atol)


    # `leastsq` has no callback option, so `states` only has before and after
    x0 = X.ravel()
    states = []
    states.append(x0.reshape(X.shape))

    x, _ = leastsq(f, x0, Dfun=Dfun, xtol=xtol, full_output=False, **kwargs)

    X = x.reshape(X.shape)
    states.append(X)

    return X, states

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

