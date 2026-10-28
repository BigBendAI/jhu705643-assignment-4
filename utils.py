"""
Module: utils.py

This module provides helper functions for video processing and data transformations 
for video classification tasks. It includes functions for:
    - Uniformly sampling frames from videos.
    - Storing extracted frames as JPEG images.
    - Retrieving image transformation statistics based on the model type.
    - Composing data transforms for training and validation/test datasets.
    - Creating DataLoaders for training, validation, and testing, using custom collate functions.
"""

import os
import cv2
import numpy as np

import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from video_datasets import collate_fn_r3d_18, collate_fn_rnn


def get_frames(vid, n_frames=1):
    """
    Uniformly sample frames from a video file.

    Args:
        vid (str): Path to the video file.
        n_frames (int): Number of frames to sample from the video.

    Returns:
        tuple: (frames, v_len)
            - frames (list): List of sampled frames (as numpy arrays in RGB format).
            - v_len (int): Number of frames that could be decoded from the video.

    Notes:
        - If the video cannot be opened or contains no frames, an empty list and 0 are returned.
        - The whole clip is decoded first and the sample positions are taken over the frames
          actually decoded. The frame count reported by the container (CAP_PROP_FRAME_COUNT) is
          an estimate and is often larger than what can be decoded, so sampling positions from it
          loses the last frame(s) of about 3% of the UCF50 clips.
        - Frames are sampled at n_frames evenly spaced positions, first and last frame included.
    """
    frames = []
    v_cap = cv2.VideoCapture(vid)
    if not v_cap.isOpened():
        print("Failed to open video:", vid)
        return frames, 0
    decoded = []
    while True:
        success, frame = v_cap.read()
        if not success:
            break
        decoded.append(frame)
    v_cap.release()
    v_len = len(decoded)
    if v_len == 0:
        print("No frames found in video:", vid)
        return frames, 0
    frame_idx = np.linspace(0, v_len - 1, n_frames).round().astype(int)
    frames = [cv2.cvtColor(decoded[idx], cv2.COLOR_BGR2RGB) for idx in frame_idx]
    return frames, v_len


def store_frames(frames, store_path):
    """
    Save a list of frames as JPEG images to the specified directory.

    Each frame is converted from RGB to BGR format (as expected by OpenCV)
    before saving.

    Args:
        frames (list): List of frames (numpy arrays in RGB format) to save.
        store_path (str): Directory path where the frames will be stored.

    Returns:
        None
    """
    os.makedirs(store_path, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Zero-padded names so that a plain sort gives the temporal order (frame_010 after 009).
        path_to_frame = os.path.join(store_path, f"frame_{idx:03d}.jpg")
        if not cv2.imwrite(path_to_frame, frame):
            raise IOError(f"Could not write frame to {path_to_frame}")


def transform_stats(model='lrcn'):
    """
    Retrieve transformation statistics based on the model type.

    For the 'lrcn' model, images are resized to 224x224; for '3dcnn', images are resized to 112x112.
    Also returns the mean and standard deviation values used for normalization.

    Args:
        model (str): Type of model ('lrcn' or '3dcnn').

    Returns:
        tuple: (h, w, mean, std)
            - h (int): Image height.
            - w (int): Image width.
            - mean (list): Mean values for normalization.
            - std (list): Standard deviation values for normalization.

    Raises:
        ValueError: If an undefined model type is provided.
    """
    if model == 'lrcn':
        h, w = 224, 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model == '3dcnn':
        h, w = 112, 112
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]
    else:
        raise ValueError('model_type arg is undefined....')
    return h, w, mean, std


def compose_data_transforms(height, width, mean, std, strong=False):
    """
    Compose and return data transforms for training and validation/test datasets.

    The training transforms include data augmentation such as random horizontal flipping and random affine transformations,
    while the validation/test transforms consist solely of resizing, converting to tensor, and normalizing.
    These are torchvision.transforms.v2 transforms applied by VideoDataset to a whole clip (a uint8
    tensor of shape (T, C, H, W)) in one call, so a clip is flipped or shifted as a whole; the
    hand-over's per-frame transforms flipped and shifted each frame on its own.

    Args:
        height (int): Desired image height.
        width (int): Desired image width.
        mean (list): Mean values for normalization.
        std (list): Standard deviation values for normalization.
        strong (bool): If True, the training transforms also take a random crop covering 70-100%
            of each side (resized back to height x width) and jitter brightness and contrast by
            up to 30%, all drawn once per clip. More variety per epoch means less memorising of
            the training clips. Default False gives the hand-over's flip and 10% translation.

    Returns:
        tuple: (train_transforms, val_test_transforms)
            - train_transforms: Composed transforms for the training set.
            - val_test_transforms: Composed transforms for the validation/test set.
    """
    if strong:
        train_transforms = v2.Compose([
            v2.RandomResizedCrop((height, width), scale=(0.49, 1.0), ratio=(0.9, 1.1),
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    else:
        train_transforms = v2.Compose([
            v2.Resize((height, width)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    val_test_transforms = v2.Compose([
        v2.Resize((height, width)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])
    return train_transforms, val_test_transforms


def make_dloader(dataset, batch_size, shuffle, collate_fn, num_workers):
    """
    Build a DataLoader that decodes clips in worker processes.

    The hand-over used the DataLoader defaults (num_workers=0), so every JPEG was decoded in the
    training process while the GPU waited; with workers the decoding overlaps with the GPU work.
    pin_memory speeds up the host-to-GPU copy.

    Args:
        dataset (Dataset): The dataset to load from.
        batch_size (int): Number of clips per batch.
        shuffle (bool): Shuffle every epoch (training only).
        collate_fn (callable): Function that turns a list of samples into a batch.
        num_workers (int): Number of loader worker processes (0 loads in the main process).

    Returns:
        DataLoader: The configured DataLoader.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0)


def train_val_dloaders(train_dataset, val_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create DataLoaders for training and validation datasets.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' (RNN-based models), uses collate_fn_rnn which pads sequences to equal lengths.
    Otherwise, uses collate_fn_r3d_18 for 3D CNN models.

    Args:
        train_dataset (Dataset): PyTorch Dataset for training data.
        val_dataset (Dataset): PyTorch Dataset for validation data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of loader worker processes.

    Returns:
        dict: Dictionary with keys 'train' and 'val' mapping to their respective DataLoaders.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    train_dl = make_dloader(train_dataset, batch_size, True, collate_fn, num_workers)
    val_dl = make_dloader(val_dataset, 2 * batch_size, False, collate_fn, num_workers)
    dataloaders = {'train': train_dl, 'val': val_dl}
    return dataloaders


def test_dloaders(test_dataset, batch_size, model='lrcn', num_workers=0):
    """
    Create a DataLoader for the test dataset.

    Selects the appropriate collate function based on the model type.
    For 'lrcn' models, uses collate_fn_rnn; otherwise, uses collate_fn_r3d_18.

    Args:
        test_dataset (Dataset): PyTorch Dataset for test data.
        batch_size (int): Number of samples per batch.
        model (str): Model type; 'lrcn' for RNN-based models, otherwise for 3D CNNs.
        num_workers (int): Number of loader worker processes.

    Returns:
        dict: Dictionary with key 'test' mapping to the test DataLoader.
    """
    collate_fn = collate_fn_rnn if model == "lrcn" else collate_fn_r3d_18
    test_dl = make_dloader(test_dataset, 2 * batch_size, False, collate_fn, num_workers)
    dataloaders = {'test': test_dl}
    return dataloaders

