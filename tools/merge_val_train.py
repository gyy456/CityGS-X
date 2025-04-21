# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
import os
import shutil


def rec_copy(from_file, to_file):
    if os.path.isdir(from_file):
        fnames = os.listdir(from_file)
        for fname in fnames:
            rec_copy(os.path.join(from_file, fname),
                     os.path.join(to_file, fname))
    else:
        shutil.copy(from_file, to_file)


def rec_remove(file):
    if os.path.isdir(file):
        fnames = os.listdir(file)
        for fname in fnames:
            rec_remove(os.path.join(file, fname))
    else:
        os.remove(file)


def copy_data(dataset_dir):
    dir_list = os.listdir(os.path.join(dataset_dir, 'val'))
    for d in dir_list:
        fnames = os.listdir(os.path.join(dataset_dir, 'val', d))
        for fname in fnames:
            rec_copy(os.path.join(dataset_dir, 'val', d, fname),
                     os.path.join(dataset_dir, 'train', d, fname))
            

def remove_data(dataset_dir):
    dir_list = os.listdir(os.path.join(dataset_dir, 'val'))
    for d in dir_list:
        fnames = os.listdir(os.path.join(dataset_dir, 'val', d))
        for fname in fnames:
            rec_remove(os.path.join(dataset_dir, 'train', d, fname))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', required=True)
    parser.add_argument('--operation', '-op', default='copy', choices=['copy', 'remove'])
    args = parser.parse_args()

    if args.operation == 'copy':
        copy_data(args.dataset_dir)
    else:
        remove_data(args.dataset_dir)
