"""
Modified from https://github.com/m-bain/frozen-in-time/blob/22a91d78405ec6032fdf521ae1ff5573358e632f/base/base_dataset.py
"""
import io
import random
import decord
import torch
import numpy as np
import math
import tarfile
from PIL import Image


decord.bridge.set_bridge("torch")


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices

    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    elif "interval" in sample:
        if num_frames == 1:
            frame_indices = [random.randint(0, vlen - 1)]
        else:
            # transform FPS
            interval = 8
            clip_length = num_frames * interval * input_fps / 30
            max_idx = max(vlen - clip_length, 0)
            start_idx = random.uniform(0, max_idx)
            end_idx = start_idx + clip_length - 1

            frame_indices = torch.linspace(start_idx, end_idx, num_frames)
            frame_indices = torch.clamp(frame_indices, 0, vlen - 1).long().tolist()
    else:
        raise ValueError
    return frame_indices


def get_frame_indices_start_end(num_frames, vlen, fps, start_time, end_time):
    start_idx = max(int(fps * start_time), 0)
    end_idx = min(int(fps * end_time), vlen)
    clip_len = end_idx - start_idx

    acc_samples = min(num_frames, clip_len)
    # split the video into `acc_samples` intervals, and sample from each interval.
    intervals = np.linspace(start=start_idx, stop=end_idx, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    
    try:
        frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
    except:
        frame_indices = np.random.permutation(list(range(start_idx, end_idx)))[:acc_samples]
        frame_indices.sort()
        frame_indices = list(frame_indices)

    if len(frame_indices) < num_frames:  # padded with last frame
        padded_frame_indices = [frame_indices[-1]] * num_frames
        padded_frame_indices[:len(frame_indices)] = frame_indices
        frame_indices = padded_frame_indices
    
    return frame_indices


def read_frames_decord(video_path, width=None, height=None, num_frames=8, sample='rand', 
    fix_start=None, max_num_frames=-1, start_time=None, end_time=None):
    if video_path.lower().endswith('.webm'):
        # a workaround for webm, large/auto num_threads will cause error.
        num_threads = 2
    else:
        num_threads = 0
    
    if width is not None and height is not None:
        video_reader = decord.VideoReader(video_path, width=width, height=height, num_threads=num_threads)
    else:
        video_reader = decord.VideoReader(video_path, num_threads=num_threads)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if start_time and end_time:
        frame_indices = get_frame_indices_start_end(
            num_frames, vlen, fps, start_time, end_time
        )
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
    frames = video_reader.get_batch(frame_indices) # TODO I dont know why it is NDarray during inference
    if isinstance(frames, torch.Tensor):
        frames=frames.numpy()  # (T, H, W, C), torch.uint8
    else:
        print(frames.shape)
        frames=frames.asnumpy()
    timestamp = {
        "num_frames": len(frame_indices),
        "timestamp": ", ".join([str(round(f / fps, 1)) for f in frame_indices])
    }
    return frames, timestamp


def read_frames_gif(video_path, num_frames=4, sample='rand'):
    video_reader = Image.open(video_path)
    vlen = video_reader.n_frames
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample
    )
    frames = []
    for idx in frame_indices:
        video_reader.seek(idx)
        frames.append(np.array(video_reader.convert('RGB')))
    frames = np.stack(frames, axis = 0) # (T, H, W, C)
    return frames
    

def read_from_tar(tar_file, bucket=None, content=None, ext_name='.mp4'):
    if content is None:
        content = './'+tar_file.split('/')[-1].replace('.tar', ext_name)
    if bucket:
        tar_file = io.BytesIO(bucket.get_object(tar_file).read())
        with tarfile.open(fileobj=tar_file) as tar:
            data = io.BytesIO(tar.extractfile(content).read())
    else:
        with tarfile.open(name=tar_file) as tar:
            data = io.BytesIO(tar.extractfile(content).read())
    return data

VIDEO_READER_FUNCS = {
    'decord': read_frames_decord
}