# solve.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import count
from misc.pickle_ import dump
from scipy.misc import imsave # requires PIL
from scipy.optimize import fmin_cg
from shard import Shard

# subaxes
def subaxes(n):
    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(float(n) / rows))
    w = 1.0 / cols
    h = 1.0 / rows

    f = plt.figure()
    axs = []
    for i in xrange(n):
        r = (rows - 1) - i / cols
        c = i % cols
        ax = f.add_axes([c * w, r * h, w, h], frameon=False)
        axs.append(ax)

    if len(axs) == 1:
        axs = axs[0]

    return f, axs

# shard_gradient
def shard_gradient(I, J, alpha, X, y, k, outside_left=True, epsilon=1e-6):
    shape = I.shape[:2]
    domain = shape[::-1]
    shard = Shard(X, k, outside_left=outside_left)

    H, dX = shard(domain, return_dX=True, epsilon=epsilon)

    d = alpha * (J - y)
    R = (I - J) + d * H[..., np.newaxis]
    J = dX[..., np.newaxis] * d
    J = J.reshape(X.size, -1).transpose()
    dX = np.dot(R.ravel(), J)
    return dX.reshape(X.shape)

# fit_shard
def fit_shard(I, J, alpha, X, y, k, outside_left=True, epsilon=1e-6, **kwargs):
    shape = I.shape[:2]
    domain = shape[::-1]

    d = alpha * (J - y)
    R0 = (I - J)

    def f(x):
        shard = Shard(x.reshape(X.shape), k, outside_left=outside_left)
        H = shard(domain)
        R = R0 + d * H[..., np.newaxis]
        r = R.ravel()
        return np.dot(r, r)

    def fprime(x, return_energy=False):
        shard = Shard(x.reshape(X.shape), k, outside_left=outside_left)
        H, dX = shard(domain, return_dX=True, epsilon=epsilon)
        R = R0 + d * H[..., np.newaxis]
        J = dX[..., np.newaxis] * d
        J = J.reshape(X.size, -1).transpose()
        r = R.ravel()
        dx = np.dot(r, J)

        if not return_energy:
            return dx
        else:
            return np.dot(r, r), dx

    callbacks = []
    def callback_handler(xk):
        for callback in callbacks:
            callback(xk)

    states = []
    def save_state(xk):
        states.append(np.copy(xk).reshape(X.shape))

    callbacks.append(save_state)

    xk = X.ravel()
    save_state(xk)

    kwargs['disp'] = 0
    try:
        callbacks.append(kwargs.pop('callback'))
    except KeyError:
        pass

    x = fmin_cg(f, xk, fprime, callback=callback_handler, **kwargs)

    return x.reshape(X.shape), states

# main_test_shard_gradient
def main_test_shard_gradient():
    INPUT_PATH = 'data/180,180-40,40,90,127,140,40-1,0,0-2.png'
    I = plt.imread(INPUT_PATH).astype(np.float64)
    shape = I.shape[:2]
    domain = shape[::-1]

    X = np.r_[40.0, 50, 100, 127, 140, 40].reshape(-1, 2)
    outside_left = True
    y = np.r_[1.0, 0.0, 0.0]
    k = 1.0
    alpha = 0.5

    # `J` is the initial image which the shard is blended with
    J = np.zeros_like(I)
    dX = shard_gradient(I, J, alpha, X, y, k, outside_left=outside_left)

    def apply_gradient(eta):
        X_ = X + eta * dX
        shard = Shard(X_, k, outside_left=outside_left)

        H = shard(domain)
        d = alpha * (J - y)
        R = (I - J) + d * H[..., np.newaxis]
        r = R.ravel()
        E = np.dot(r, r)
        print '%.7g -> %.7g' % (eta, E)
        return H

    etas = np.arange(20) * -0.05
    f, axs = subaxes(etas.shape[0])

    for i, eta in enumerate(etas):
        H = apply_gradient(eta)
        Ji = (J * (1.0 - alpha * H[..., np.newaxis]) 
              + alpha * y * H[..., np.newaxis])
        axs[i].imshow(Ji)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()

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
    # main_test_shard_gradient()
    main()
    
