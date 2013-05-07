# shard.py

# Imports
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
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
    def __init__(self, m, x0):
        self.m = m
        self.x0 = x0

    @classmethod
    def from_points(cls, *args):
        if len(args) == 1:
            p, q = args[0]
        else:
            p, q = args
        return cls(q - p, p)

    @property
    def points(self):
        return np.r_['0,2', self.x0, self.x0 + self.l * self.m]

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        m = np.asarray(m)
        norm_m = norm(m)
        self._m = m / norm_m
        self.l = norm_m

    @property
    def n(self):
        return np.r_[-self.m[1], self.m[0]]

    def __call__(self, t):
        t = np.atleast_1d(t)
        return self.m * self.l * t[:, np.newaxis] + self.x0

    # closest_preimage
    def closest_preimage(self, Q):
        Q = np.atleast_2d(Q)
        u = np.atleast_1d(np.dot(Q - self.x0, self.m) / self.l)
        u[u < 0.0] = 0.0
        u[u > 1.0] = 1.0
        return u

    # dt
    def dt(self, integral_domain, outside_left=None):
        # create `X` as scanning `integral_domain` along rows, from
        # "top" (max y) to "bottom" (min y)
        slice_ = tuple(slice(None, d) for d in integral_domain[::-1])
        G = map(np.ravel, np.mgrid[slice_])
        X = np.transpose(G)
        X = np.fliplr(X)
        X[:,1] *= -1
        X[:,1] += integral_domain[1] 

        u = self.closest_preimage(X)
        Y = self(u)
        r = X - Y
        d = np.sqrt(np.sum(r**2, axis=1))

        if outside_left is not None:
            s = (np.dot(r, self.n) > 0).astype(np.float64)
            s *= 2.0
            s -= 1.0
            if not outside_left:
                s *= -1.0
            d *= s
                
        return d.reshape(integral_domain[::-1])

    def __repr__(self):
        fmt = "[%.7g, %.7g]"
        return "%s(%s, %s, %.7g)" % (self.__class__.__name__,
                                     fmt % tuple(self._m),
                                     fmt % tuple(self._x0),
                                     self._l)

# Polygon
class Polygon(object):
    def __init__(self, lines):
        self.lines = list(lines)

    # is_closed
    def is_closed(self, tolerance):
        for i, line in enumerate(self.lines):
            prev_line = self.lines[i - 1]
            end = prev_line.points[1]
            start = line.points[0]
            if norm(end - start) > tolerance:
                return False

        return True
        
    @classmethod
    def from_points(cls, P, *args, **kwargs):
        if len(P) < 3:
            raise ValueError("number of points = %d (< 3)" % P.shape[0])

        lines = []
        for i, p in enumerate(P):
            q = P[(i + 1) % len(P)]
            lines.append(LineSegment.from_points(p, q))

        return cls(lines, *args, **kwargs)

    @property
    def points(self):
        return np.asarray(map(lambda l: l.points[0], 
                              self.lines))

    @points.setter
    def points(self, P):
        if len(self.lines) != len(P):
            raise ValueError("len(P) != %d" % len(self.lines))
            
        for i, p in enumerate(P):
            q = P[(i + 1) % len(P)]
            self.lines[i].x0 = p
            self.lines[i].m = q - p

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           repr(self.lines))

    # dt
    def dt(self, integral_domain, outside_left=None):
        D = np.asarray(map(lambda l: l.dt(integral_domain, outside_left),
                           self.lines))
        D = D.transpose(1, 2, 0)
        shape = D.shape
        D = D.reshape(-1, shape[2])

        if outside_left is not None:
            s = np.any(D > 0.0, axis=1).astype(np.float64)
            s *= 2.0 
            s -= 1.0

        np.absolute(D, D)
        I = np.argmin(D, axis=1)
        D = D[np.arange(D.shape[0]), I]

        if outside_left is not None:
            D *= s

        return I.reshape(shape[:2]), D.reshape(shape[:2])

# sigmoid
def sigmoid(t, k=1.0):
    t = np.asarray(t)
    return 1.0 / (1.0 + np.exp(k * t))

# Shard
class Shard(object):
    def __init__(self, X, k, outside_left=True):
        self._X = np.atleast_2d(X)
        self._k = k
        self._outside_left = outside_left

        self._poly = Polygon.from_points(self._X)

    def __call__(self, integral_domain, return_dX=False, epsilon=1e-6):
        I, D = self._poly.dt(integral_domain, self._outside_left)
        H = sigmoid(D, self._k)
        if not return_dX:
            return H

        _x = self._X.ravel()
        n = _x.shape[0]

        dX = np.empty((n,) + H.shape, dtype=np.float64)
        for i in xrange(n):
            x = _x.copy()
            x[i] += epsilon
            self._poly.points = x.reshape(self._X.shape)
            dX[i] = self(integral_domain)

        dX -= H
        dX /= epsilon
        return H, dX

# main_test_LineSegment
def main_test_LineSegment():
    x0 = np.r_[0.0, 0.0]
    m = np.r_[2.0, 1.0]
    m *= (4.0 / norm(m))
    line = LineSegment(m, x0)

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

    D = line.dt((50, 100), outside_left=True)
    f, ax = plt.subplots()
    ax.imshow(D)
    plt.show()

# main_test_Polygon
def main_test_Polygon():
    P1 = np.array([[  10.,   10.],
                   [ 135.,   60.],
                   [  60.,   10.]])
    P2 = P1.copy()
    P2[1] += (10.0, 40.0)

    domain = (160, 100)

    poly = Polygon.from_points(P1)

    for P in (P1, P2):
        print 'is closed?', poly.is_closed(1e-4)
        poly.points = P

        I, D = poly.dt(domain, outside_left=True)

        x, y = np.transpose(np.r_['0,2', P, P[0]])
        y = D.shape[0] - y

        to_view = (D, sigmoid(D, k=1.0), I)
        f, axs = plt.subplots(len(to_view), 1)
        for ax, M in zip(axs, to_view):
            ax.imshow(M)
            ax.set_xlim(0, M.shape[1] - 1)
            ax.set_ylim(M.shape[0] - 1, 0)
            ax.plot(x, y, 'ro-')

    plt.show()

# main_test_Shard
def main_test_Shard():
    P = np.array([[  10.,   10.],
                  [ 135.,   60.],
                  [  60.,   10.]])
    k = 0.6

    shard = Shard(P, k, outside_left=True)
    H, dX = shard((150, 100), return_dX=True, epsilon=1e-6)

    # colour `dX` so that all images are on the same scale
    min_, max_ = np.amin(dX), np.amax(dX)
    scaled_dX = (dX - min_) * (255. / (max_ - min_))
    I = np.around(scaled_dX).astype(np.int32)
    cmap = cm.gray(np.linspace(0., 1., 256, endpoint=True))
    coloured_dX = cmap[I]

    assert dX.shape[0] % 2 == 0
    f, axs = plt.subplots(2, dX.shape[0] / 2)
    for i, D in enumerate(dX):
        ax = axs[i % 2, i / 2]
        ax.set_title('x[%d] : (%.5g, %.5g)' % (i, np.amin(D), np.amax(D)))
        ax.imshow(coloured_dX[i])
    plt.show()

if __name__ == '__main__':
    # main_test_LineSegment()
    # main_test_linedt()
    # main_test_Polygon()
    main_test_Shard()


