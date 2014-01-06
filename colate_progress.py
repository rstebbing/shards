# colate_progress.py

# Imports
import argparse
import matplotlib.pyplot as plt
import os
import subprocess

from itertools import dropwhile
from operator import add
from pprint import pprint
from shutil import copy

# image_number
def image_number(filename):
    root = os.path.splitext(filename)[0]
    try:
        return int(root)
    except ValueError:
        return None

# iterations_in_shard_subdir
def iterations_in_shard_subdir(dir_):
    image_files = filter(lambda f: os.path.splitext(f)[1] == '.png',
                         os.listdir(dir_))
    numbered_images = filter(lambda f: image_number(f) is not None,
                             image_files)
    sorted_image_files = sorted(numbered_images, key=image_number)
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

    # TODO Put `temp_path` into a context manager?
    temp_path = os.path.join(head, root + '_temp')
    if os.path.exists(temp_path):
        raise RuntimeError('temporary path "%s" exists' % temp_path)
    if temp_path and not os.path.exists(temp_path):
        os.makedirs(temp_path)

    for image_paths, output_path in [(iteration_image_paths, args.output_path),
                                     (final_image_paths, final_output_path)]:
        if args.repeat_frames > 1:
            duplicated_paths = []
            for path in image_paths:
                duplicated_paths += [path] * args.repeat_frames
            image_paths = duplicated_paths


        # Copy images to `temp_path`.
        temp_image_paths = []
        for i, image_path in enumerate(image_paths):
            temp_image_path = os.path.join(temp_path, '%d.png' %  i)
            temp_image_paths.append(temp_image_path)
            if os.path.exists(temp_image_path):
                continue
            print '%s -> %s' % (image_path, temp_image_path)
            copy(image_path, temp_image_path)

        h, w = plt.imread(image_paths[0]).shape[:2]

        cmd = ['ffmpeg',
               '-i', os.path.join(temp_path, '%d.png'),
               '-vcodec', 'mpeg4',
               output_path]
        print ' '.join(cmd)
        subprocess.check_call(cmd)

        for temp_image_path in temp_image_paths:
            print '! %s' % temp_image_path
            os.remove(temp_image_path)

    os.rmdir(temp_path)

if __name__ == '__main__':
    main()
