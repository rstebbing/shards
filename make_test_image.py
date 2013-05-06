# make_test_image.py

# Imports
import argparse
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from shard import Shard

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    parser.add_argument('k', type=float)
    parser.add_argument('positions', type=float, nargs='*')
    parser.add_argument('--output-path', default='{default}')
    parser.add_argument('--outside-right', 
                        dest='outside_left',
                        action='store_false',
                        default=True)
    args = parser.parse_args()

    if len(args.positions) % 2 != 0:
        raise ValueError("positions must be specified in pairs")

    if '{default}' in args.output_path:
        domain_string = '%d,%d' % (args.width, args.height)
        positions_string = ','.join('%.3g' % p for p in args.positions)
        components = [domain_string, '%.3g' % args.k, positions_string]
        default = '%s.png' % '-'.join(components)
        output_path = args.output_path.format(default=default)
    else:
        output_path = args.output_path

    X = np.asarray(args.positions).reshape(-1, 2)
    shard = Shard(X, 1.0, args.k, outside_left=args.outside_left)
    
    domain = (args.width, args.height)
    D = shard(domain)
    print '->', output_path
    plt.imsave(output_path, D, cmap=cm.gray)

if __name__ == '__main__':
    main()
