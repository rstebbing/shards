##########################################
# File: make_test_image.py               #
# Copyright Richard Stebbing 2014.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from shard import Shard
from scipy.misc import imsave # requires PIL

# float_ndarray
def float_ndarray(s):
    return np.asarray(map(float, s.split(',')))

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    parser.add_argument('y', type=float_ndarray)
    parser.add_argument('k', type=float)
    parser.add_argument('positions', type=float, nargs='*')
    parser.add_argument('--output-path', default='{default}')
    args = parser.parse_args()

    if len(args.y) != 3:
        raise ValueError("y must be a 3-tuple")
    if np.any(args.y < 0.0) or np.any(args.y > 1.0):
        raise ValueError("0 <= y[i] <= 1")

    if len(args.positions) % 2 != 0:
        raise ValueError("positions must be specified in pairs")

    if '{default}' in args.output_path:
        float_string = lambda l: ','.join(map(lambda f: '%.3g' % f, l))
        components = ['%d,%d' % (args.width, args.height),
                      float_string(args.positions),
                      float_string(args.y),
                      float_string([args.k])]
        default = '%s.png' % '-'.join(components)
        output_path = args.output_path.format(default=default)
    else:
        output_path = args.output_path

    X = np.asarray(args.positions).reshape(-1, 2)
    shard = Shard(X, args.k)

    domain = (args.width, args.height)
    alpha = shard(domain)
    D = args.y.reshape(-1, 1, 1) * alpha
    D = np.around(D * 255.0).astype(np.uint8)

    print '->', output_path
    imsave(output_path, D)

if __name__ == '__main__':
    main()
