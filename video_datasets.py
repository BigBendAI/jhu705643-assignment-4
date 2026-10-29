"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
by person group into cross-validation folds and save/load a split as a CSV, and the collate
functions that turn a list of clips into a batch.
"""

import os
import re
import csv
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from tqdm import tqdm


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (list): List of (video directory path, integer label) pairs.
        fr_per_vid (int): Number of frames per video to load, in time order.
        transforms (callable, optional): A transform applied to the whole clip at once, a uint8
            tensor of shape (T, C, H, W). torchvision.transforms.v2 transforms treat the clip as
            a batch of images and draw their random parameters once, so every frame of a clip
            gets the same flip and shift.
        random_clip (bool, optional): How to pick fr_per_vid frames out of the stored frames.
            False: evenly spaced positions, the same every time (validation and test). True: the
            stored frames are cut into fr_per_vid equal segments and one frame is drawn at random
            from each segment every time the clip is loaded, so each epoch trains on a different
            clip of the same video (training). Default is False.
        view (int, optional): With random_clip False, which of n_views interleaved evenly spaced
            clips of the video to take: view 0 starts at the first stored frame, view k starts
            k / n_views of a step later. Predict mode averages the predictions over all views,
            as the paper averages over clips of a video. Default is 0.
        n_views (int, optional): Number of interleaved clips. Default is 4.
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None, random_clip=False, view=0,
                 n_views=4):
        if not 0 <= view < n_views:
            raise ValueError('view must be between 0 and n_views - 1')
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.transforms = transforms
        self.random_clip = random_clip
        self.view = view
        self.n_views = n_views

    def frame_indices(self, n_stored):
        """
        Positions of the fr_per_vid frames to load out of n_stored stored frames.

        Args:
            n_stored (int): Number of frame files in the clip directory.

        Returns:
            ndarray: fr_per_vid indices in increasing order.
        """
        if n_stored <= self.fpv or not (self.random_clip or self.view):
            # Evenly spaced; repeats when fewer frames are stored, so every clip has the length.
            return np.linspace(0, n_stored - 1, self.fpv).round().astype(int)
        if not self.random_clip:
            # The same spacing, started view / n_views of a step later (never past the last frame).
            positions = (np.linspace(0, n_stored - 1, self.fpv)
                         + self.view * (n_stored - 1) / (self.n_views * self.fpv))
            return np.minimum(positions.round().astype(int), n_stored - 1)
        # One random frame per segment. torch's random generator is seeded per DataLoader
        # worker, so the workers do not all draw the same frames.
        edges = np.linspace(0, n_stored, self.fpv + 1).astype(int)
        offsets = torch.rand(self.fpv).numpy() * (edges[1:] - edges[:-1])
        return edges[:-1] + offsets.astype(int)

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load fr_per_vid frames of the video at the given index in time order, apply the clip
        transform, and return the clip tensor along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T = fr_per_vid and label is an integer.
        """
        # The frame files in time order. glob returns directory order, which is arbitrary, so the
        # hand-over fed the LSTM shuffled frames; the names are zero-padded, so sorting fixes it.
        fr_paths = sorted(glob.glob(os.path.join(self.dataset[idx][0], '*.jpg')))
        if not fr_paths:
            raise FileNotFoundError(f'No frames found in {self.dataset[idx][0]}')

        # fr_per_vid of the stored frames (see frame_indices), decoded into one uint8 tensor
        fr_idx = self.frame_indices(len(fr_paths))
        fr_imgs = torch.stack([decode_image(fr_paths[i], mode=ImageReadMode.RGB) for i in fr_idx])

        # Transform the whole clip at once (same random flip/shift for all of its frames)
        if self.transforms:
            fr_imgs = self.transforms(fr_imgs)

        return fr_imgs, self.dataset[idx][1]


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
    Collate function for the LRCN.

    Every clip from VideoDataset has the same number of frames, so the clips are simply stacked
    into a tensor of shape (batch_size, T, C, H, W) next to a tensor of labels. (The hand-over
    padded variable-length clips and silently dropped empty ones; VideoDataset now raises on an
    empty clip directory instead.)

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked clips, shape (batch_size, T, C, H, W).
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = zip(*batch)
    return torch.stack(imgs_batch), torch.tensor(label_batch)
