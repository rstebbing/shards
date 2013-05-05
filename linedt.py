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

# main
def main():
    x0 = np.r_[0.0, 0.0]
    m = np.r_[2.0, 1.0]
    m /= norm(m)
    l = 4.0

    def M(t):
        t = np.atleast_1d(t)
        return m * l * t[:, np.newaxis] + x0

    def M_inv(Q):
        Q = np.atleast_2d(Q)
        u = np.atleast_1d(np.dot(Q - x0, m) / l)
        u[u < 0.0] = 0.0
        u[u > 1.0] = 1.0
        return u

    Q = np.array([[2.0, 2.0],
                  [4.0, 3.0]], dtype=np.float64)
    u = M_inv(Q)

    f, ax = plt.subplots()
    ax.set_aspect('equal')
    plot = Plotter(delta=0.05)

    t = np.linspace(0.0, 1.0, 20, endpoint=True)
    plot(ax, M(t), 'ro-')

    for i, ui in enumerate(u):
        plot(ax, np.r_['0,2', Q[i], M(ui)], 'bo-')

    ax.set_xlim(plot.xlim)
    ax.set_ylim(plot.ylim)

    plt.show()

if __name__ == '__main__':
    main()
