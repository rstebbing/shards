# make_test_image.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from shard import Shard
from scipy.misc import imsave # requires PIL

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    parser.add_argument('y', type=str)
    parser.add_argument('k', type=float)
    parser.add_argument('positions', type=float, nargs='*')
    parser.add_argument('--output-path', default='{default}')
    parser.add_argument('--outside-right', 
                        dest='outside_left',
                        action='store_false',
                        default=True)
    args = parser.parse_args()

    args.y = eval(args.y)
    if len(args.y) != 3:
        raise ValueError("y must be a 3-tuple")

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
    shard = Shard(X, args.y, args.k, outside_left=args.outside_left)
    
    domain = (args.width, args.height)
    D = shard(domain).transpose(1, 2, 0)
    D = np.around(D * 255.0).astype(np.uint8)

    print '->', output_path
    imsave(output_path, D)

if __name__ == '__main__':
    main()
