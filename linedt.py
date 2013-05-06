# linedt.py

# Imports
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import norm

# Plotter
class Plotter(object):
    def __init__(self, delta=0.0):
        self._delta = delta

        i = np.finfo(np.float64)
        self._xmin = i.max
        self._xmax = i.min
        self._ymin = i.max
        self._ymax = i.min

    def __call__(self, ax, P, *args, **kwargs):
        P = np.atleast_2d(P)
        x, y = np.transpose(P)
        ax.plot(x, y, *args, **kwargs)

        self._xmin = min(self._xmin, np.amin(x))
        self._xmax = max(self._xmax, np.amax(x))
        self._ymin = min(self._ymin, np.amin(y))
        self._ymax = max(self._ymax, np.amax(y))

    def _lim(self, min_, max_):
        d = self._delta * (max_ - min_)
        return (min_ - d, max_ + d)

    @property
    def xlim(self):
        return self._lim(self._xmin, self._xmax)

    @property
    def ylim(self):
        return self._lim(self._ymin, self._ymax)

# LineSegment
class LineSegment(object):
    def __init__(self, m, x0, l = 1.0):
        norm_m = norm(m)
        self._m = m / norm_m
        self._x0 = x0
        self._l = l * norm_m
        self._n = np.r_[-self._m[1], self._m[0]]

    def __call__(self, t):
        t = np.atleast_1d(t)
        return self._m * self._l * t[:, np.newaxis] + self._x0

    def closest_preimage(self, Q):
        Q = np.atleast_2d(Q)
        u = np.atleast_1d(np.dot(Q - self._x0, self._m) / self._l)
        u[u < 0.0] = 0.0
        u[u > 1.0] = 1.0
        return u

    @property
    def m(self):
        return self._m

    @property
    def n(self):
        return self._n

# linedt
def linedt(line, integral_domain, outside_left=None):
    # create `X` as scanning `integral_domain` along rows, from
    # "top" (max y) to "bottom" (min y)
    slice_ = tuple(slice(None, d) for d in integral_domain[::-1])
    G = map(np.ravel, np.mgrid[slice_])
    X = np.transpose(G)
    X = np.fliplr(X)
    X[:,1] *= -1
    X[:,1] += integral_domain[1] 

    u = line.closest_preimage(X)
    Y = line(u)
    r = X - Y
    d = np.sqrt(np.sum(r**2, axis=1))

    if outside_left is not None:
        s = (np.dot(r, line.n) > 0).astype(np.float64)
        s *= 2.0
        s -= 1.0
        if not outside_left:
            s *= -1.0
        d *= s
            
    return d.reshape(integral_domain[::-1])

# main_test_LineSegment
def main_test_LineSegment():
    x0 = np.r_[0.0, 0.0]
    m = np.r_[2.0, 1.0]
    m /= norm(m)
    l = 4.0
    line = LineSegment(m, x0, l)

    Q = np.array([[2.0, 2.0],
                  [4.0, 3.0]], dtype=np.float64)
    u = line.closest_preimage(Q)

    f, ax = plt.subplots()
    ax.set_aspect('equal')
    plot = Plotter(delta=0.05)

    t = np.linspace(0.0, 1.0, 20, endpoint=True)
    plot(ax, line(t), 'ro-')

    for i, ui in enumerate(u):
        plot(ax, np.r_['0,2', Q[i], line(ui)], 'bo-')

    ax.set_xlim(plot.xlim)
    ax.set_ylim(plot.ylim)

    plt.show()

# main_test_linedt
def main_test_linedt():
    x0 = np.r_[25.0, 25.0]
    m = np.r_[2.0, 1.0]
    m *= (10.0 / norm(m))
    line = LineSegment(m, x0)

    D = linedt(line, (50, 100), outside_left=True)
    f, ax = plt.subplots()
    ax.imshow(D)
    plt.show()

if __name__ == '__main__':
    # main_test_LineSegment()
    main_test_linedt()

