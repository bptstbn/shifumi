"""
contains some utilities like
- drawing samples from a dataset
-
"""
import os
import random
import shutil


def sample_ds(
        source=os.path.join('datasets', 'custom'),
        dest=os.path.join('datasets', 'custom_sample'),
        no_of_files=20,
        subdirs=['rock', 'paper', 'scissors'],
        copy=False):
    """
    moves a sample of n files from each subdir, sampled randomly

    for creating a test set, set copy to false so that files are moved

    for creating a validation set, set copy to true so that files are NOT moved

    :param source: the source of the dataset
    :param dest:  the destination of the sample
    :param no_of_files: how many samples should be drawn
    :param subdirs: the subdirs from which you want to sample
    :param copy: if set true the files are copies, otherwise files are moved
    """
    for subdir in subdirs:
        source_sub = os.path.join(source, subdir)
        dest_sub = os.path.join(dest, subdir)
        if not os.path.exists(dest_sub):
            os.makedirs(dest_sub)
        files = os.listdir(source_sub)
        for file_name in random.sample(files, no_of_files):
            if copy:
                shutil.copy(os.path.join(source_sub, file_name), dest_sub)
            else:
                shutil.move(os.path.join(source_sub, file_name), dest_sub)


def rename_files_in_dir(dir=os.getcwd(), leading_text=""):
    """
    renames all files in a directory (xyz.abc -> {leading_text}__xyz.abc)

    :param dir: the directory to be renamed
    :param leading_text: the leading text
    """
    if not os.path.isdir(dir):
        return
    for f in [f.path for f in os.scandir(dir) if not f.is_dir()]:
        f_name, f_ext = os.path.splitext(f)
        f_name_wo_dir = f_name.split('/')[-1]
        new_name = os.path.join(dir, f'{leading_text}__{f_name_wo_dir}{f_ext}')
        os.rename(f, new_name)


def rename_entire_ds(dir=os.getcwd(), leading_text=""):
    """
    renames an entire ds bc recursively calling rename files in dir

    :param dir: path to dataset
    :param leading_text: the new leading text
    """
    if not os.path.isdir(dir):
        return
    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for s in subdirs:
        # For each folder call recursively
        rename_entire_ds(s, leading_text)
    rename_files_in_dir(dir, leading_text)

def rename_all_datasets_with_int (dir=os.path.join('datasets', 'combine')):
    """
    renames all files for datasets in a folder with a dataset specific leading number
    
    :param dir: directory in which the datasets reside 
    """
    for cnt, ds in enumerate(os.scandir(dir)):
        rename_entire_ds(ds, str(cnt))
