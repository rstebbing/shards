# solve.py

# Imports
import numpy as np

from itertools import count, ifilter, imap
from operator import mul
from scipy.optimize import approx_fprime, leastsq
from shard import Shard, sigmoid, sigmoid_dt, inverse_sigmoid

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

# fit_and_colour_shard
def fit_and_colour_shard(I, J, alpha, X, y, k, epsilon=1e-6, xtol=1e-4,
                         check_gradients=False,
                         **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    def structure_x(X, y):
        return np.r_[X.ravel(), inverse_sigmoid(y)]
    def destructure_x(x, return_t=False):
        X_, t = x[:-y.size].reshape(X.shape), x[-y.size:]
        y_ = sigmoid(t)
        return (X_, y_, t) if return_t else (X_, y_)

    R0 = (I - J)
    def f(x):
        X_, y_ = destructure_x(x)
        shard = Shard(X_, k)
        H = shard(domain)
        R = R0 + alpha * (J - y_) * H[..., np.newaxis]
        return R.ravel()

    def Dfun(x):
        X_, y_, t_ = destructure_x(x, return_t=True)
        shard = Shard(X_, k)
        H, dX = shard(domain, return_dX=True, epsilon=epsilon)
        d = alpha * (J - y_)
        JX = dX[..., np.newaxis] * d
        aH = -alpha * H
        dy = sigmoid_dt(t_)
        n = y.size
        Jy = np.zeros(((n,) + H.shape + (n,)), dtype=np.float64)
        for i in xrange(n):
            Jy[i, ..., i] = dy[i] * aH
        return np.c_[JX.reshape(X.size, -1).T, Jy.reshape(n, -1).T]

    if check_gradients:
        # set y < 1.0 - epsilon for forward difference used by `approx_fprime`
        y1 = np.copy(y)
        max_y = 1.0 - 2 * epsilon
        y1[y1 > max_y] = max_y
        x = structure_x(X, y1)
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
    states = []
    def save_state(x):
        X_, y_ = destructure_x(x)
        states.append((X_, y_))

    x0 = structure_x(X, y)
    save_state(x0)

    x, _ = leastsq(f, x0, Dfun=Dfun, xtol=xtol, full_output=False, **kwargs)
    save_state(x)

    X, y = states[-1]
    return X, y, states

# fit_and_colour_shards
def fit_and_colour_shards(I, J0, alpha, Xs, ys, k, epsilon=1e-6,
                          ftol=1e-8, xtol=1e-8, maxfev=0,
                          check_gradients=False, return_info=False,
                          verbose=False,
                          **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    Xs = np.require(np.atleast_2d(Xs), dtype=np.float64)
    ys = np.require(np.atleast_1d(ys), dtype=np.float64)
    X_shape, X_size, y_size = Xs[0].shape, Xs[0].size, ys[0].size
    N = len(Xs)
    x_size = (X_size + y_size) * N

    if maxfev == 0:
        # same as `leastsq` but used by verbose option in `f(x)`
        maxfev = 100 * (x_size + 1)

    def structure_x(Xs, ys):
        return np.hstack(map(np.ravel, Xs) + map(inverse_sigmoid, ys))
    def destructure_x(x, return_ts=False):
        Xs = list(x[:N * X_size].reshape((N,) + X_shape))
        ts = x[N* X_size:].reshape(N, y_size)
        ys = list(sigmoid(ts))
        return (Xs, ys, ts) if return_ts else (Xs, ys)

    def build_J(x):
        Xs, ys = destructure_x(x)
        J = J0.copy()
        for i in xrange(N):
            shard = Shard(Xs[i], k)
            aH = alpha * shard(domain)
            J += (ys[i] - J) * aH[..., np.newaxis]
        return J

    fx_eval_count = count(1)
    def f(x):
        R = I - build_J(x)
        r = R.ravel()
        if verbose:
            # ugh
            print ' [%d/%d]: %g' % (next(fx_eval_count), maxfev,
                                    0.5 * np.dot(r, r))
        return r
    def e(x):
        r = f(x)
        return 0.5 * np.dot(r, r)

    def prod(I, A, start=0, l=None):
        N = len(A)
        indices = xrange(start, N)
        if l is not None:
            indices = ifilter(lambda i: i != l, indices)
        As = imap(A.__getitem__, indices)
        return reduce(mul, As, I)
    def Dfun(x):
        Xs, ys, ts = destructure_x(x, return_ts=True)
        aHs, omaHs, adHs = [], [], []
        for i in xrange(N):
            shard = Shard(Xs[i], k)
            H, dX = shard(domain, return_dX=True, epsilon=epsilon)
            aH = alpha * H
            aHs.append(aH)
            omaHs.append(1.0 - aH)
            adHs.append(alpha * dX)

        I = np.ones_like(aHs[0])
        JXs, Jts = [], []
        for l in xrange(N):
            JXl = J0 * prod(I, omaHs, 0, l)[..., np.newaxis]
            for i in xrange(l):
                JXli = aHs[i] * prod(I, omaHs, i + 1, l)
                JXl += ys[i] * JXli[..., np.newaxis]
            prod_lp1 = prod(I, omaHs, l + 1)
            JXl -= ys[l] * prod_lp1[..., np.newaxis]
            JXl_T = adHs[l][..., np.newaxis] * JXl
            JXs.append(JXl_T.reshape(X_size, -1))

            neg_aH_prod_lp1 = -(aHs[l] * prod_lp1)
            dy = sigmoid_dt(ts[l])
            Jt = np.zeros(((y_size,) + I.shape + (y_size,)),
                          dtype=np.float64)
            for i in xrange(y_size):
                Jt[i, ..., i] = dy[i] * neg_aH_prod_lp1
            Jts.append(Jt.reshape(y_size, -1))
        return np.vstack(JXs + Jts).T

    if check_gradients:
        x = structure_x(Xs, ys)
        approx_D = approx_fprime(x, e, epsilon=epsilon)
        J_ = Dfun(x)
        r = f(x)
        D = np.dot(r, J_)
        print 'approx_D: (%4g, %4g)' % (np.amin(approx_D), np.amax(approx_D))
        print 'D: (%4g, %4g)' % (np.amin(D), np.amax(D))
        atol = 1e-4
        print 'allclose (atol=%g)?' % atol, np.allclose(approx_D, D, atol=atol)

    states = []
    def save_state(x):
        states.append(destructure_x(x))

    x0 = structure_x(Xs, ys)
    save_state(x0)

    x, exit_code = leastsq(f, x0, Dfun=Dfun,
                           ftol=ftol, xtol=xtol, maxfev=maxfev,
                           full_output=False, **kwargs)
    save_state(x)

    Xs, ys = states[-1]
    if return_info:
        ei, Ji = e(x0), build_J(x0)
        ef, Jf = e(x), build_J(x)
        return (Xs, ys, states), (exit_code, ei, ef, Ji, Jf)
    else:
        return Xs, ys, states

