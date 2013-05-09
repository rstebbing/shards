# visualise_progress.py

# Imports
import argparse
import errno
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.linalg import norm

try:
    from scipy.misc import imsave # requires PIL
except ImportError:
    from matplotlib.pyplot import imsave

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

# make_visualisations_inplace
def make_visualisations_inplace(all_X, J1s, output_dir, verbose=False):
    for i, im in enumerate(J1s):
        f = plt.figure()
        ax = f.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        ax.imshow(im)

        X = all_X[i]
        x, y = np.transpose(np.r_['0,2', X, X[0]])
        y = im.shape[0] - y

        ac = np.mean(im.reshape(-1, im.shape[-1]))
        d_to_black = norm(ac)
        d_to_white = norm(1.0 - ac)
        c = 'w' if d_to_black < d_to_white else 'k'

        ax.plot(x, y, '-', c=c)

        ax.set_xlim(-0.5, im.shape[1] - 0.5)
        ax.set_ylim(im.shape[0] - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        output_path = os.path.join(output_dir, '%d.png' % i)
        print '->', output_path

        dpi = 100
        f.set_dpi(dpi)
        size = np.asarray(im.shape[:2][::-1], dtype=np.float64) / dpi
        f.set_size_inches(size)
        f.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)

    output_path = os.path.join(output_dir, '-1.png')
    print '->', output_path
    imsave(output_path, np.around(J1s[-1] * 255.0).astype(np.uint8))

