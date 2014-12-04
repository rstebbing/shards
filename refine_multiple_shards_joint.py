##########################################
# File: refine_multiple_shards_joint.py  #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import visualise_progress as vis

from functools import partial
from operator import itemgetter
from solve import fit_and_colour_shards
from time import time

# Requires `rscommon`.
from rscommon.pickle_ import dump

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_all_iterations_Xy_path')
    parser.add_argument('output_dir')
    parser.add_argument('--visualise-progress',
                        action='store_true',
                        default=False)
    parser.add_argument('--ftol', type=float, default=1e-8)
    parser.add_argument('--xtol', type=float, default=1e-8)
    parser.add_argument('--maxfev', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    args = parser.parse_args()

    ensure_output_path = partial(vis.ensure_path, args.output_dir)

    all_iterations_Xy, orig_args = np.load(args.input_all_iterations_Xy_path)
    print '<-', orig_args['input_path']
    I = plt.imread(orig_args['input_path']).astype(np.float64)[..., :3]

    if orig_args['base'] == 'white':
        J0 = np.ones_like(I)
    elif orig_args['base'] == 'black':
        J0 = np.zeros_like(I)
    else:
        head, tail = os.path.split(orig_args['base'])
        root, ext = os.path.splitext(tail)
        if ext == '.dat':
            J0 = np.load(orig_args['base'])
        else:
            J0 = plt.imread(orig_args['base']).astype(np.float64)[..., :3]

    Xs0, ys0 = zip(*map(itemgetter(-1), all_iterations_Xy))
    print 'Solving with `fit_and_colour_shards` ...'
    np.seterr(over='ignore')
    t0 = time()
    (Xs, ys, all_Xs_ys), (exit_code, E0, E1, J, J1) = fit_and_colour_shards(
        I, J0, orig_args['alpha'],
        Xs0, ys0,
        k=orig_args['k'],
        epsilon=args.epsilon,
        ftol=args.ftol,
        xtol=args.xtol,
        maxfev=args.maxfev,
        return_info=True,
        verbose=True)
    t1 = time()
    np.seterr(over='warn')
    print 'E0:', E0
    print 'E1:', E1
    print 'Exit code: %d' % exit_code
    print 'Time taken: %.3fs' % (t1 - t0)

    output_path = ensure_output_path('all_Xs_ys.dat')
    print '->', output_path
    dump(output_path, (all_Xs_ys, args.__dict__), raise_on_failure=False)

    output_path = ensure_output_path('J.dat')
    print '->', output_path
    dump(output_path, (J, args.__dict__), raise_on_failure=False)

    output_path = ensure_output_path('J1.dat')
    print '->', output_path
    dump(output_path, (J1, args.__dict__), raise_on_failure=False)

    if args.visualise_progress:
        output_path = ensure_output_path('J.png')
        print '->', output_path
        f, ax = vis.make_image_figure(J)
        vis.save_image_figure(output_path, f, J.shape)

        output_path = ensure_output_path('J1.png')
        print '->', output_path
        f, ax = vis.make_image_figure(J1)
        vis.save_image_figure(output_path, f, J1.shape)

if __name__ == '__main__':
    main()
