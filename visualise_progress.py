# visualise_progress.py

# Imports
import argparse
import errno
import matplotlib.pyplot as plt
import numpy as np
import os

from operator import itemgetter
from reconstruct import ShardReconstructor
from scipy.linalg import norm

# ensure_path
def ensure_path(dir_, *p, **kwargs):
    is_dir = kwargs.get('is_dir', False)

    full_path = os.path.join(dir_, *map(str, p))
    if is_dir:
        head = full_path
    else:
        head, tail = os.path.split(full_path)

    try:
        os.makedirs(head)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise

    return full_path

# make_image_figure
def make_image_figure(im):
    f = plt.figure(frameon=False)
    ax = f.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    ax.imshow(im, interpolation='none')
    return f, ax

# save_image_figure
def save_image_figure(output_path, f, shape):
    DPI = 72
    ax = f.axes[0]
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.itervalues():
        spine.set_visible(False)

    f.set_dpi(DPI)
    size = np.asarray(shape[:2][::-1], dtype=np.float64) / DPI
    f.set_size_inches(size)
    f.savefig(output_path, dpi=DPI,
              bbox_inches='tight', pad_inches=0.0,
              frameon=False)
    plt.close(f)

# make_visualisations_inplace
def make_visualisations_inplace(all_X, J1s, output_dir, verbose=False):
    for i, im in enumerate(J1s):
        f, ax = make_image_figure(im)

        X = all_X[i]
        x, y = np.transpose(np.r_['0,2', X, X[0]])
        y = im.shape[0] - y

        ac = np.mean(im.reshape(-1, im.shape[-1]))
        d_to_black = norm(ac)
        d_to_white = norm(1.0 - ac)
        c = 'w' if d_to_black < d_to_white else 'k'

        ax.plot(x, y, '-', c=c)

        output_path = os.path.join(output_dir, '%d.png' % i)
        if verbose:
            print '->', output_path

        save_image_figure(output_path, f, im.shape)

    output_path = os.path.join(output_dir, '-1.png')
    if verbose:
        print '->', output_path

    f, ax = make_image_figure(J1s[-1])
    save_image_figure(output_path, f, J1s[-1].shape)

# make_residual_image
def make_residual_image(I, J, output_dir, verbose=False):
    def save(filename, A):
        output_path = os.path.join(output_dir, filename)
        if verbose:
            print '->', output_path
        f, ax = make_image_figure(A)
        save_image_figure(output_path, f, A.shape)

    R = I - J
    save('R.png', R)

    Rsq = np.power(R, 2)
    save('Rsq.png', Rsq)

    E = np.sum(Rsq, axis=-1)
    save('E.png', E)

    for i in xrange(Rsq.shape[2]):
        save('Rsq_%d.png' % i, Rsq[..., i])

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path', nargs='?', default=None)
    args = parser.parse_args()

    def ensure_output_directory(*p):
        root = args.output_path
        if root is None:
            root = args.input_path

        return ensure_path(os.path.join(root, *p),
                           is_dir=True)

    walker = os.walk(args.input_path)
    input_dir, shard_subdirs, files = next(walker)
    shard_subdirs = sorted(shard_subdirs, key=int)
    shard_dirs = map(lambda d: os.path.join(input_dir, d),
                     shard_subdirs)
    state_paths = map(lambda d: os.path.join(d, 'all_Xy.dat'),
                      shard_dirs)
    all_Xy, a = np.load(state_paths[0])

    print '<-', a['input_path']
    I = plt.imread(a['input_path']).astype(np.float64)[..., :3]

    if a['base'] == 'white':
        J = np.ones_like(I)
    elif a['base'] == 'black':
        J = np.zeros_like(I)
    else:
        head, tail = os.path.split(a['base'])
        root, ext = os.path.splitext(tail)
        if ext == '.dat':
            J = np.load(a.base)
        else:
            J = plt.imread(a.base).astype(np.float64)[..., :3]

    sr = ShardReconstructor(I, a['alpha'], n=a['n'], k=a['k'])

    for state_index, state_path in enumerate(state_paths):
        print '<-', state_path
        all_Xy, _ = np.load(state_path)

        J1s = map(lambda t: sr.add_shard_to_reconstruction(J, t[0], t[1]),
                  all_Xy)
        output_directory = ensure_output_directory(shard_subdirs[state_index])

        make_visualisations_inplace(map(itemgetter(0), all_Xy),
                                    J1s,
                                    output_directory,
                                    verbose=True)
        make_residual_image(I, J1s[-1], output_directory, verbose=True)

        J = np.load(os.path.join(shard_dirs[state_index], 'J.dat'))

if __name__ == '__main__':
    main()

