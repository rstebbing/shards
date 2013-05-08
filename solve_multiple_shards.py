# solve_multiple_shards.py

# Imports
import argparse
import errno
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import count
from misc.pickle_ import dump
from sample import sample_polygon
from scipy.misc import imsave # requires PIL
from shard import Shard, Polygon
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

            # force `X` to respect `outside_left = True`
            X = np.flipud(X)

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
        shard = Shard(X, self._k, outside_left=True)
        H = shard(self._domain)
        J1 = (J * (1.0 - self._alpha * H[..., np.newaxis])
              + self._alpha * y * H[..., np.newaxis])
        J1[J1 > 1.0] = 1.0
        J1[J1 < 0.0] = 0.0
        return J1

    def new_reconstruction(self, base):
        if base == 'black':
            return np.zeros_like(self._I)
        else:
            return np.ones_like(self._I)
            
    def candidate_shard(self, J, maxiter=10, verbose=False, **kwargs):
        X = self._sample_shard()
        y = self._colour_shard(J, X)
        if verbose:
            print 'X:'
            print X
            print 'y:', y

        callback=None

        if verbose:
            print 'solving ...'
            solver_iteration = count(1)
            def print_solver_iteration(xk):
                print ' %d/%d' % (next(solver_iteration), maxiter)

            callback = print_solver_iteration

        kwargs = kwargs.copy()
        kwargs.setdefault('maxiter', maxiter)
        kwargs['callback'] = callback

        X1, all_X = fit_shard(self._I, J, self._alpha, 
                              X, y, self._k,
                              outside_left=True,
                              **kwargs)

        all_y = map(lambda X: y, all_X)

        y1 = self._colour_shard(J, X1)
        all_X.append(X1)
        all_y.append(y1)

        return X1, y1, zip(all_X, all_y)

    def reconstruction_energy(self, J):
        r = (self._I - J).ravel()
        return np.dot(r, r)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_dir')
    parser.add_argument('base', type=str, choices=('black', 'white'))
    parser.add_argument('k', type=float)
    parser.add_argument('alpha', type=float)
    parser.add_argument('shards', type=int)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def ensure_output_path(*p):
        full_path = os.path.join(args.output_dir, *map(str, p))

        head, tail = os.path.split(full_path)
        try:
            os.makedirs(head)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise

        return full_path

    print '<-', args.input_path
    I = plt.imread(args.input_path).astype(np.float64)[..., :3]

    sr = ShardReconstructor(I, args.alpha, n=args.n, k=args.k)
    J = sr.new_reconstruction(args.base)

    for n in xrange(args.shards):
        X, y, all_Xy = sr.candidate_shard(J, maxiter=args.maxiter,
                                          verbose=True)

        J1s = map(lambda t: sr.add_shard_to_reconstruction(J, t[0], t[1]), 
                  all_Xy)

        for i, im in enumerate(J1s):
            f = plt.figure()
            ax = f.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

            im = np.around(im * 255.0).astype(np.uint8)
            ax.imshow(im)

            X = all_Xy[i][0]
            x, y = np.transpose(np.r_['0,2', X, X[0]])
            y = im.shape[0] - y
            ax.plot(x, y, '-', c='w' if args.base == 'black' else 'k')

            ax.set_xlim(0, im.shape[1] - 1)
            ax.set_ylim(im.shape[0] - 1, 0)
            ax.set_xticks([])
            ax.set_yticks([])

            output_path = ensure_output_path(n, '%d.png' % i)
            print '->', output_path
            dpi = 100
            f.set_dpi(dpi)
            size = np.asarray(im.shape[:2][::-1], dtype=np.float64) / dpi
            f.set_size_inches(size)
            f.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
            plt.close(f)
            
        J = J1s[-1]

if __name__ == '__main__':
    main()
