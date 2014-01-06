# reconstruct.py

# Imports
import numpy as np

from itertools import count
from sample import sample_polygon
from shard import Polygon, Shard
from solve import fit_shard, colour_shard

# ShardReconstructor
class ShardReconstructor(object):
    def __init__(self, I, alpha, n=3, k=1.0):
        self._I = I
        self._alpha = alpha
        self._n = n
        self._k = k
        self._shape = I.shape[:2]
        self._domain = self._shape[::-1]

    def _colour_shard(self, J, X):
        return colour_shard(self._I, J, self._alpha, X, self._k,
                            limit_colours=True)

    def _sample_shard(self):
        while True:
            X = sample_polygon(self._n)
            X *= np.random.rand() * np.amin(self._domain)
            X += np.random.rand(len(self._domain)) * self._domain
            
            max_, min_ = np.amax(X, axis=0), np.amin(X, axis=0)
            required_displacement = np.empty(len(max_), dtype=np.float64)
            for i, d in enumerate(self._domain):
                if max_[i] > d:
                    required_displacement[i] = d - max_[i]
                elif min_[i] < 0:
                    required_displacement[i] = -min_[i]
                else:
                    required_displacement[i] = 0.0

            X += required_displacement

            try:
                poly = Polygon.from_points(X)
            except ZeroDivisionError:
                pass
            else:
                return X

    def add_shard_to_reconstruction(self, J, X, y):
        shard = Shard(X, self._k)
        H = shard(self._domain)
        J1 = (J * (1.0 - self._alpha * H[..., np.newaxis])
              + self._alpha * y * H[..., np.newaxis])
        J1[J1 > 1.0] = 1.0
        J1[J1 < 0.0] = 0.0
        return J1

    def candidate_shard(self, J, verbose=False, **kwargs):
        X = self._sample_shard()
        y = self._colour_shard(J, X)
        if verbose:
            print 'X:'
            print X
            print 'y:', y

        X1, all_X = fit_shard(self._I, J, self._alpha,
                              X, y, self._k,
                              **kwargs)

        all_Xy = map(lambda x: (x, y), all_X)

        y1 = self._colour_shard(J, X1)
        all_Xy.append((X1, y1))

        return X1, y1, all_Xy

    def reconstruction_energy(self, J):
        r = (self._I - J).ravel()
        return np.dot(r, r)

