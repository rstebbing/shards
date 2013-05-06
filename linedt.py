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
        m = np.asarray(m)
        norm_m = norm(m)
        self._m = m / norm_m
        self._x0 = x0
        self._l = l * norm_m
        self._n = np.r_[-self._m[1], self._m[0]]
        self._x1 = self._x0 + self._l * self._m
        self._points = np.r_['0,2', self._x0, self._x1]

    @classmethod
    def from_points(cls, *args):
        if len(args) == 1:
            p, q = args[0]
        else:
            p, q = args
        return cls(q - p, p)

    @property
    def points(self):
        return self._points

    def __call__(self, t):
        t = np.atleast_1d(t)
        return self._m * self._l * t[:, np.newaxis] + self._x0

    # closest_preimage
    def closest_preimage(self, Q):
        Q = np.atleast_2d(Q)
        u = np.atleast_1d(np.dot(Q - self._x0, self._m) / self._l)
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
            s = (np.dot(r, self._n) > 0).astype(np.float64)
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
    def __init__(self, lines, tolerance=1e-4):
        self._lines = []
        for i, line in enumerate(lines):
            prev_line = lines[i - 1]
            end = prev_line.points[1]
            start = line.points[0]
            if norm(end - start) > tolerance:
                raise ValueError(
                    "%s does not meet %s with tolerance %.3g (%s != %s)" % 
                    (prev_line, line, tolerance, end, start))

            self._lines.append(line)

    @classmethod
    def from_points(cls, P, *args, **kwargs):
        P = np.atleast_2d(P)
        if P.shape[0] < 3:
            raise ValueError("number of points = %d (< 3)" % P.shape[0])

        lines = []
        for i, p in enumerate(P):
            q = P[(i + 1) % len(P)]
            lines.append(LineSegment.from_points(p, q))

        return cls(lines, *args, **kwargs)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           repr(self._lines))

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

    D = line.dt((50, 100), outside_left=True)
    f, ax = plt.subplots()
    ax.imshow(D)
    plt.show()

# main_test_Polygon
def main_test_Polygon():
    poly = Polygon.from_points([(0.0, 0.0),
                                (0.5, 0.5 * np.sqrt(3.0)),
                                (1.0, 0.0)])
    print poly

if __name__ == '__main__':
    # main_test_LineSegment()
    # main_test_linedt()
    main_test_Polygon()

