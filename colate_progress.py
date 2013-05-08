# colate_progress.py

# Imports
import argparse
import matplotlib.pyplot as plt
import os 
import subprocess

from itertools import dropwhile
from operator import add
from pprint import pprint

# iterations_in_shard_subdir
def iterations_in_shard_subdir(dir_):
    image_files = filter(lambda f: os.path.splitext(f)[1] == '.png',
                         os.listdir(dir_))
    image_number = lambda f: int(os.path.splitext(f)[0])
    sorted_image_files = sorted(image_files, key=image_number)
    valid_image_files = list(dropwhile(lambda f: image_number(f) < 0,
                             sorted_image_files))
    return map(lambda f: os.path.join(dir_, f), valid_image_files)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    parser.add_argument('--repeat-frames', type=int, default=1)
    args = parser.parse_args()

    walker = os.walk(args.input_path)
    input_dir, shard_subdirs, files = next(walker)
    shard_subdirs = sorted(shard_subdirs, key=int)
    shard_dirs = map(lambda d: os.path.join(input_dir, d),
                     shard_subdirs)
    iteration_image_paths = reduce(
        add, map(iterations_in_shard_subdir, shard_dirs))
    final_image_paths = map(lambda d: os.path.join(input_dir, d, '-1.png'),
                            shard_subdirs)

    head, tail = os.path.split(args.output_path)
    root, ext = os.path.splitext(tail)
    final_output_path = os.path.join(head, root + '_-1' + ext)

    for image_paths, output_path in [(iteration_image_paths, args.output_path),
                                     (final_image_paths, final_output_path)]:
        if args.repeat_frames > 1:
            duplicated_paths = []
            for path in image_paths:
                duplicated_paths += [path] * args.repeat_frames
            image_paths = duplicated_paths

        listing_path = os.path.join(args.input_path, 'images.txt')
        print '%d images ->' % len(image_paths), listing_path
        with open(listing_path, 'w') as fp:
            fp.write('\n'.join(image_paths))

        h, w = plt.imread(image_paths[0]).shape[:2]

        cmd = ['mencoder',
               'mf://@%s' % listing_path,
               '-mf',
               'w=%d:h=%d:fps=25:type=png' % (w, h),
               '-ovc',
               'lavc', '-lavcopts', 'vcodec=mpeg4:vbitrate=15000:mbd=2',
               '-o', output_path]
        print ' '.join(cmd)
        subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
