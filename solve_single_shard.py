# solve_single_shard.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import count
from misc.pickle_ import dump
from scipy.misc import imsave # requires PIL
from solve import fit_shard
from shard import Shard

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_dir')
    parser.add_argument('base', type=str, choices=('black', 'white'))
    parser.add_argument('y', type=str)
    parser.add_argument('k', type=float)
    parser.add_argument('alpha', type=float)
    parser.add_argument('positions', type=float, nargs='*')
    parser.add_argument('--outside-right', 
                        dest='outside_left',
                        action='store_false',
                        default=True)
    parser.add_argument('--maxiter', type=int, default=10)

    args = parser.parse_args()
    args.y = np.asarray(eval(args.y))

    if len(args.y) != 3:
        raise ValueError("y must be a 3-tuple")
    if np.any(args.y < 0.0) or np.any(args.y > 1.0):
        raise ValueError("0 <= y[i] <= 1")

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("0 <= alpha <= 1")

    if len(args.positions) % 2 != 0:
        raise ValueError("positions must be specified in pairs")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    make_output_path = lambda f: os.path.join(args.output_dir, f)

    print '<-', args.input_path
    I = plt.imread(args.input_path).astype(np.float64)
    shape = I.shape[:2]
    domain = shape[::-1]

    X = np.asarray(args.positions).reshape(-1, 2)
    J = np.empty_like(I)
    if args.base == 'black':
        J.fill(0.)
    else:
        J.fill(1.)

    print 'X:'
    print X
    print 'y:', args.y
    print 'k:', args.k
    print 'alpha:', args.alpha
    print 'outside_left:', args.outside_left

    solver_iteration = count(1)
    def print_solver_iteration(xk):
        print ' %d/%d' % (next(solver_iteration), args.maxiter)

    print 'solving ...'
    X1, all_X = fit_shard(I, J, args.alpha, X, args.y, args.k, 
                          outside_left=args.outside_left,
                          maxiter=args.maxiter,
                          callback=print_solver_iteration)
    output_path = make_output_path('X.dat')
    print '->', output_path
    dump(output_path, (X1, all_X))

    def X_to_image(X):
        shard = Shard(X, args.k, outside_left=args.outside_left)
        H = shard(domain)
        Ji = (J * (1.0 - args.alpha * H[..., np.newaxis]) 
              + args.alpha * args.y * H[..., np.newaxis])
        return Ji
    all_images = map(X_to_image, all_X)

    for i, im in enumerate(all_images):
        output_path = make_output_path('%d.png' % i)
        print '->', output_path
        im = np.around(im * 255.0).astype(np.uint8)
        imsave(output_path, im)

    output_path = make_output_path('I.png')
    print '->', output_path
    imsave(output_path, np.around(I * 255.0).astype(np.uint8))

if __name__ == '__main__':
    main()

        
