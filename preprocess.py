"""
Module: preprocess.py

Convert every video of the dataset into a folder of evenly spaced JPEG frames. The README of the
original code base refers to this script, but it was not part of the hand-over; the two helpers it
builds on (``get_frames`` and ``store_frames`` in utils.py) were.

The frame folders mirror the class folders of the dataset:

    <video_dir>/<Class>/<clip>.avi   ->   <frame_dir>/<Class>/<clip>/frame_000.jpg ... frame_031.jpg

Usage:
    python preprocess.py --video_dir data/UCF50 --frame_dir data/frames --n_frames 32 --workers 8
"""

import argparse
import os
from multiprocessing import Pool

from tqdm import tqdm

from utils import get_frames, store_frames


def list_videos(video_dir):
    """
    List all videos below ``video_dir`` as (class name, video path) pairs, sorted.

    Args:
        video_dir (str): Directory with one sub-directory per class containing the video files.

    Returns:
        list: (class_name, video_path) tuples.
    """
    videos = []
    for class_name in sorted(os.listdir(video_dir)):
        class_dir = os.path.join(video_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(('.avi', '.mp4')):
                videos.append((class_name, os.path.join(class_dir, file_name)))
    return videos


def extract_one(job):
    """
    Extract and store the frames of one video (worker function for the process pool).

    Args:
        job (tuple): (video_path, clip_dir, n_frames).

    Returns:
        tuple: (video_path, number of frames stored).
    """
    video_path, clip_dir, n_frames = job
    frames, _ = get_frames(video_path, n_frames)
    store_frames(frames, clip_dir)
    return video_path, len(frames)


def main():
    """Parse the arguments and extract the frames of every video, skipping clips already done."""
    parser = argparse.ArgumentParser(description='Extract evenly spaced frames from every video')
    parser.add_argument('--video_dir', required=True, help='Dataset root with one folder per class')
    parser.add_argument('--frame_dir', required=True, help='Output root for the frame folders')
    parser.add_argument('--n_frames', type=int, default=32,
                        help='Frames to keep per video (the model uses 16 of them per clip)')
    parser.add_argument('--workers', type=int, default=8, help='Parallel worker processes')
    args = parser.parse_args()

    videos = list_videos(args.video_dir)
    jobs = []
    for class_name, video_path in videos:
        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_dir = os.path.join(args.frame_dir, class_name, clip_name)
        if os.path.isdir(clip_dir) and len(os.listdir(clip_dir)) == args.n_frames:
            continue  # already extracted
        jobs.append((video_path, clip_dir, args.n_frames))
    print(f'{len(videos)} videos found, {len(jobs)} to extract')

    short = []
    with Pool(args.workers) as pool:
        for video_path, n_stored in tqdm(pool.imap_unordered(extract_one, jobs), total=len(jobs)):
            if n_stored < args.n_frames:
                short.append((video_path, n_stored))
    print(f'done: {len(jobs)} videos extracted to {args.frame_dir}')
    if short:
        print(f'{len(short)} videos gave fewer than {args.n_frames} frames:')
        for video_path, n_stored in short:
            print(f'  {video_path} ({n_stored} frames)')


if __name__ == '__main__':
    main()
