# solve_multiple_shards.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import visualise_progress as vis

from functools import partial
from operator import itemgetter
from pickle_ import dump
from reconstruct import ShardReconstructor
from time import time

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_dir')
    parser.add_argument('base', help='{white, black, BASE}')
    parser.add_argument('k', type=float)
    parser.add_argument('alpha', type=float)
    parser.add_argument('shards', type=int)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=10)
    parser.add_argument('--update-colours', 
                        action='store_true',
                        default=False)
    parser.add_argument('--visualise-progress', 
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    ensure_output_path = partial(vis.ensure_path, args.output_dir)

    print '<-', args.input_path
    I = plt.imread(args.input_path).astype(np.float64)[..., :3]

    if args.base == 'white':
        J = np.ones_like(I)
    elif args.base == 'black':
        J = np.zeros_like(I)
    else:
        head, tail = os.path.split(args.base)
        root, ext = os.path.splitext(tail)
        if ext == '.dat':
            J = np.load(args.base)
        else:
            J = plt.imread(args.base).astype(np.float64)[..., :3]

    sr = ShardReconstructor(I, args.alpha, n=args.n, k=args.k)

    t1 = time()

    for n in xrange(args.shards):
        X, y, all_Xy = sr.candidate_shard(J, maxiter=args.maxiter,
                                          update_colours=args.update_colours,
                                          verbose=True)

        J1s = map(lambda t: sr.add_shard_to_reconstruction(J, t[0], t[1]), 
                  all_Xy)

        output_dir = ensure_output_path(n, is_dir=True)

        if args.visualise_progress:
            vis.make_visualisations_inplace(map(itemgetter(0), all_Xy),
                                            J1s,
                                            output_dir)
                
        J = J1s[-1]
            
        output_path = ensure_output_path(n, 'all_Xy.dat')
        print '->', output_path
        dump(output_path, (all_Xy, args.__dict__))

        output_path = ensure_output_path(n, 'J.dat')
        print '->', output_path
        dump(output_path, J)

        print '%d shards in %.3fs' % (n + 1, time() - t1)

if __name__ == '__main__':
    main()
