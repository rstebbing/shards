# colate_progress.py

# Imports
import argparse
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
    parser.add_argument('--delay', type=int, default=10)
    args = parser.parse_args()

    walker = os.walk(args.input_path)
    input_dir, shard_subdirs, files = next(walker)
    shard_subdirs = sorted(shard_subdirs, key=int)
    shard_dirs = map(lambda d: os.path.join(input_dir, d),
                     shard_subdirs)
    image_paths = reduce(add, map(iterations_in_shard_subdir, shard_dirs))

    image_list_file = '_IMAGES.txt'
    print '%d images ->' % len(image_paths), image_list_file
    with open(image_list_file, 'w') as fp:
        fp.write('\n'.join(image_paths))

    cmd = ['convert',
           '-delay', str(args.delay),
           '-loop', '0',
           '@%s' % image_list_file,
           args.output_path]
    print ' '.join(cmd)
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
