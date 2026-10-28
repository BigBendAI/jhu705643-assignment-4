"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
by person group into cross-validation folds and save/load a split as a CSV, and custom collate
functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import re
import csv
import glob
from tqdm import tqdm
import numpy as np


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.
    
    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.
    
    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load (images are taken in order).
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Get all JPEG frame paths from the video directory and select up to fr_per_vid frames
        fr_paths = glob.glob(self.dataset[idx][0] + '/*.jpg')
        fr_paths = fr_paths[:self.fpv]
        
        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]
        
        # Get the label associated with the video
        fr_label = self.dataset[idx][1]
        
        # Apply transforms to each frame if provided, else keep original images
        fr_imgs_trans = [self.transforms(fr_img) for fr_img in fr_imgs] if self.transforms else fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.
    
    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.
    
    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.
    
    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(sorted(os.listdir(frame_dir)))}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(sorted(os.listdir(frame_dir))):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            vid_path = os.path.join(vid_cat_path, vid)
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def clip_group(vid_path):
    """
    Return the (class name, group number) of a clip from its folder name.

    UCF50 clip folders are named like v_Basketball_g07_c03: g07 is the group. The clips of one
    group are cut from the same long video, so they show the same person, background and
    viewpoint. If one clip of a group is in the training set, the other clips of that group are
    almost memorised, so a group must never be shared between train, validation and test.

    Args:
        vid_path (str): Path to the clip's frame directory.

    Returns:
        tuple: (class name, group number).
    """
    match = re.match(r'v_(.+)_g(\d+)_c\d+$', os.path.basename(vid_path))
    if match is None:
        raise ValueError(f'Cannot read the group from clip folder name {vid_path!r}')
    return match.group(1), int(match.group(2))


def dataset_split(vid_dataset, fold, n_folds=5, n_val_groups=2, seed=0):
    """
    Split the dataset into training, validation and test sets by group, for one fold.

    Within each class the groups (see clip_group) are shuffled once with the seed and dealt
    round-robin into n_folds folds. The groups of the requested fold form the test set, the
    next n_val_groups of the remaining groups form the validation set and all other groups form
    the training set. Every clip of a group lands in the same set, every class is in every set,
    and running every fold in turn tests the model on every group once (cross-validation).

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        fold (int): Which fold is the test set (0 .. n_folds - 1).
        n_folds (int, optional): Number of folds. Default is 5 (about 5 test groups per class).
        n_val_groups (int, optional): Validation groups per class. Default is 2.
        seed (int, optional): Random seed for the shuffle. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset), each a list of (video_path, label) tuples.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f'fold must be between 0 and {n_folds - 1}')
    print(f'Splitting train/validation/test datasets by group (fold {fold} of {n_folds})....')
    clips_by_group = {}
    for clip in vid_dataset.items():          # clip = (video path, label)
        clips_by_group.setdefault(clip_group(clip[0]), []).append(clip)

    rng = np.random.default_rng(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for class_name in sorted({name for name, _ in clips_by_group}):
        groups = sorted(group for name, group in clips_by_group if name == class_name)
        groups = [groups[i] for i in rng.permutation(len(groups))]
        test_groups = groups[fold::n_folds]
        rest = [group for group in groups if group not in test_groups]
        for group in groups:
            if group in test_groups:
                target = 'test'
            elif group in rest[:n_val_groups]:
                target = 'val'
            else:
                target = 'train'
            splits[target].extend(sorted(clips_by_group[(class_name, group)]))
    return splits['train'], splits['val'], splits['test']


def split_overlap(splits):
    """
    Return the (class, group) pairs that appear in more than one of the splits.

    Args:
        splits (dict): Dictionary with keys 'train', 'val' and 'test', each a list of
            (video_path, label) tuples.

    Returns:
        set: The shared (class name, group number) pairs; empty when the split is leak-free.
    """
    groups = {name: {clip_group(vid_path) for vid_path, _ in clips}
              for name, clips in splits.items()}
    return ((groups['train'] & groups['val']) | (groups['train'] & groups['test'])
            | (groups['val'] & groups['test']))


def write_split_csv(csv_path, splits, frame_dir):
    """
    Save a split as a CSV with the columns split, clip_dir, label.

    clip_dir is stored relative to frame_dir, so the CSV can be committed and used on another
    machine where the frames live in a different directory.

    Args:
        csv_path (str): Where to write the CSV.
        splits (dict): Dictionary with keys 'train', 'val' and 'test' (lists of (path, label)).
        frame_dir (str): The frame directory the paths are relative to.
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['split', 'clip_dir', 'label'])
        for name in ('train', 'val', 'test'):
            for vid_path, vid_label in splits[name]:
                writer.writerow([name, os.path.relpath(vid_path, frame_dir), int(vid_label)])


def read_split_csv(csv_path, frame_dir):
    """
    Load a split written by write_split_csv, turning clip_dir back into a full path.

    Args:
        csv_path (str): The CSV to read.
        frame_dir (str): The frame directory to join clip_dir onto.

    Returns:
        dict: Dictionary with keys 'train', 'val' and 'test', each a list of (path, label).
    """
    splits = {'train': [], 'val': [], 'test': []}
    with open(csv_path, newline='', encoding='utf-8') as csv_file:
        for row in csv.DictReader(csv_file):
            vid_path = os.path.join(frame_dir, row['clip_dir'])
            splits[row['split']].append((vid_path, int(row['label'])))
    return splits


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).
    
    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.
    
    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.
    
    Args:
        batch (list): List of samples, each as (video_frames, label).
    
    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))
    
    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)
    
    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)
    
    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)
    
    return padded_imgs, labels_tensor
