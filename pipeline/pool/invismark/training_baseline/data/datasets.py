import glob
import logging
import os
import threading
from collections import OrderedDict

from decord import VideoReader, cpu
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    """A PyTorch Dataset for loading video clips from a folder of MP4 files."""

    def __init__(
        self,
        folder_path: str,
        frames_per_clip: int = 2,  # Number of frames in each video clip
        frame_step: int = 24,  # Step size between frames within a clip
        transform: callable = None,
        # numbers of cpu to run the preprocessing of each batch
        num_workers: int = 8,
        buffer_size: int = 15,
    ):
        """Initializes the VideoDataset.

        Args:
            folder_path: Path to the folder containing video files organized
                in subdirectories.
            frames_per_clip: Number of frames to extract per video clip.
            frame_step: Step size between consecutive frames within a clip.
            transform: Optional transform to apply to each frame.
            num_workers: Number of CPU threads for video decoding.
            buffer_size: Maximum number of videos to cache in memory.
        """
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.transform = transform
        self.num_workers = num_workers
        self.videofiles = np.array(
            glob.glob(os.path.join(folder_path, '*', '*.mp4')))
        logger.info("Found %d videos in %s", len(self.videofiles), folder_path)
        self.video_buffer = LRUDict(maxsize=buffer_size)

    def __getitem__(self, index):
        """Returns a video clip at the given index.

        Args:
            index: Index of the video file to load.

        Returns:
            A tuple of (clip, label) where clip is a tensor of shape
            (frames_per_clip, channels, height, width) and label is 0.
        """
        video_file = self.videofiles[index]
        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer = self.load_clip(video_file)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                logger.info(
                    f"Failed to load video {video_file}. Trying another video.")
                videofile_index = np.random.randint(self.__len__())
                video_file = self.videofiles[videofile_index]
        # rearrange to (num_clips, frames_per_clip, channels, height, width)
        buffer = torch.from_numpy(buffer).permute(
            0, 3, 1, 2).unsqueeze(0).float()

        if self.transform is None:
            return buffer
        clip = torch.stack([self.transform(frame) for frame in buffer[0]])
        return clip, 0

    def load_clip(self, fname):
        """Loads a clip of frames from a video file.

        Args:
            fname: Path to the video file.

        Returns:
            A numpy array of shape (frames_per_clip, H, W, 3) with values
            normalized to [-1, 1], or an empty list if loading fails.
        """
        if fname in self.video_buffer:
            vr = self.video_buffer[fname]
        else:
            try:
                vr = VideoReader(
                    fname, num_threads=self.num_workers, ctx=cpu(0))
            except Exception:
                return [], None
            self.video_buffer[fname] = vr

        sample_range = len(vr) - (self.frames_per_clip-1)*self.frame_step
        if sample_range <= 0:
            return [], None

        start_indx = np.random.randint(0, sample_range)
        indices = np.array([start_indx + self.frame_step * i
                            for i in range(self.frames_per_clip)])
        buffer = vr.get_batch(indices).asnumpy()
        # Normalized value between -1 and 1.
        buffer = 2.0 * buffer / 255.0 - 1.0
        return buffer

    def __len__(self):
        """Returns the total number of videos in the dataset."""
        return len(self.videofiles)


def video_train_dataloader(video_path, batch_size, frame_step, num_workers=0):
    """Creates a DataLoader for video training with distributed sampling.

    Args:
        video_path: Path to the folder containing video files.
        batch_size: Number of samples per batch.
        frame_step: Step size between frames within each clip.
        num_workers: Number of worker processes for data loading.

    Returns:
        A DataLoader configured for distributed training.
    """
    dataset = VideoDataset(video_path, transform=transforms.Compose(
        [transforms.Resize((512, 512)), ]), frame_step=frame_step)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset))


def video_eval_dataloader(video_path,
                          batch_size=1,
                          frames_per_clip=24,
                          frame_step=1,
                          num_workers=0):
    """Creates a DataLoader for video evaluation.

    Args:
        video_path: Path to the folder containing video files.
        batch_size: Number of samples per batch.
        frames_per_clip: Number of frames to extract per video clip.
        frame_step: Step size between frames within each clip.
        num_workers: Number of worker processes for data loading.

    Returns:
        A DataLoader configured for evaluation with shuffling enabled.
    """
    dataset = VideoDataset(
        video_path, frames_per_clip=frames_per_clip, frame_step=frame_step)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)


def img_train_dataloader(path, batch_size, num_workers=8):
    """Creates a DataLoader for image training with distributed sampling.

    Args:
        path: Path to the image folder organized by class subdirectories.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A DataLoader configured for distributed training with images
        resized to 256x256 and normalized to [-1, 1].
    """
    dataset = dset.ImageFolder(root=path, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        # scale image pixels from [0, 255] to [0, 1] values
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset)
    )


def img_eval_dataloader(path, batch_size, num_workers=8):
    """Creates a DataLoader for image evaluation.

    Args:
        path: Path to the image folder organized by class subdirectories.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A DataLoader configured for evaluation with shuffling enabled
        and images normalized to [-1, 1].
    """
    dataset = dset.ImageFolder(root=path, transform=transforms.Compose([
        transforms.ToTensor(),
        # scale image pixels from [0, 255] to [0, 1] values
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)


class LRUDict(OrderedDict):
    """A thread-safe Least Recently Used (LRU) dictionary with a maximum size."""

    def __init__(self, maxsize=10):
        """Initializes the LRUDict.

        Args:
            maxsize: Maximum number of items to store before eviction.
        """
        super().__init__()
        self.maxsize = maxsize
        self.lock = threading.RLock()  # Use a reentrant lock to avoid deadlocks

    def __setitem__(self, key, value):
        """Sets an item in the dictionary, evicting LRU items if needed.

        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        with self.lock:
            # Insert the item in the dictionary
            super().__setitem__(key, value)

            # If the dictionary exceeds max size, remove the least recently used items
            if len(self) > self.maxsize:
                self._cleanup()

    def __getitem__(self, key):
        """Gets an item from the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key is not found.
        """
        with self.lock:
            value = super().__getitem__(key)
            # Move the accessed item to the end to mark it as recently used
            return value

    def __delitem__(self, key):
        """Deletes an item from the dictionary.

        Args:
            key: The key to delete.

        Raises:
            KeyError: If the key is not found.
        """
        with self.lock:
            super().__delitem__(key)

    def _cleanup(self):
        """Removes least recently used items to maintain size limit.

        Clears 10% of the maxsize or at least 1 item.
        """
        # Remove the least recently used items until we're back under the limit
        # Clear 10% or at least 1
        num_to_clear = max(1, int(0.1 * self.maxsize))
        for _ in range(num_to_clear):
            self.popitem(last=False)  # Remove from the start (LRU)
