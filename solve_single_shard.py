# solve_single_shard.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import count
from pickle_ import dump
from solve import fit_shard, colour_shard
from shard import Shard

try:
    from scipy.misc import imsave # requires PIL
except ImportError:
    from matplotlib.pyplot import imsave

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_dir')
    parser.add_argument('base', type=str, choices=('black', 'white'))
    parser.add_argument('k', type=float)
    parser.add_argument('alpha', type=float)
    parser.add_argument('positions', type=float, nargs='*')
    parser.add_argument('--no-limit-colours', 
                        dest='limit_colours',
                        action='store_false',
                        default=True)
    parser.add_argument('--y', type=str, default='None')
    parser.add_argument('--maxiter', type=int, default=10)

    args = parser.parse_args()
    args.y = eval(args.y)

    if args.alpha < 0.0 or args.alpha > 1.0:
        raise ValueError("0 <= alpha <= 1")

    if len(args.positions) % 2 != 0:
        raise ValueError("positions must be specified in pairs")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    make_output_path = lambda f: os.path.join(args.output_dir, f)

    print '<-', args.input_path
    I = plt.imread(args.input_path).astype(np.float64)[..., :3]
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
    print 'k:', args.k
    print 'alpha:', args.alpha

    if args.y is None:
        print 'initialise y (limit_colours = "%s") ...' % args.limit_colours
        y = colour_shard(I, J, args.alpha, X, args.k, 
                         limit_colours=args.limit_colours)
    else:
        y = np.asarray(args.y)
        
    print 'y:', y

    solver_iteration = count(1)
    def print_solver_iteration(xk):
        print ' %d/%d' % (next(solver_iteration), args.maxiter)

    print 'solving X ...'
    X1, all_X = fit_shard(I, J, args.alpha, X, y, args.k,
                          maxiter=args.maxiter,
                          callback=print_solver_iteration)
    all_y = map(lambda X: y, all_X)
    print 'X1:'
    print X1

    print 'solving y (limit_colours = "%s") ...' % args.limit_colours
    y1 = colour_shard(I, J, args.alpha, X1, args.k, 
                      limit_colours=args.limit_colours)
    print 'y1:', y1

    all_X.append(X1)
    all_y.append(y1)

    output_path = make_output_path('X.dat')
    print '->', output_path
    dump(output_path, (X1, all_X, all_y))

    def X_to_image(X, y):
        shard = Shard(X, args.k)
        H = shard(domain)
        Ji = (J * (1.0 - args.alpha * H[..., np.newaxis]) 
              + args.alpha * y * H[..., np.newaxis])
        Ji[Ji > 1.0] = 1.0
        Ji[Ji < 0.0] = 0.0
        return Ji
    all_images = map(X_to_image, all_X, all_y)

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

        
