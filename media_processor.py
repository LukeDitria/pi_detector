import os
from pathlib import Path
import cv2
from typing import Union, Generator, List, Optional
import numpy as np


def process_frame(frame, image_wh, crop_to_square):
    lores_wh = image_wh
    h, w, c = frame.shape
    if crop_to_square:
        lores_w = int(round(image_wh[0] * (w / h)))
        lores_wh = (lores_w, image_wh[1])

    frame = cv2.resize(frame, lores_wh)

    return frame

class VideoProcessor:
    # Common video file extensions
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

    def __init__(self, path: Union[str, Path], image_wh: tuple[int, int], crop_to_square: bool):
        """
        Initialize the MediaProcessor with a path to a directory or a video file.

        Args:
            path: Path to a directory containing images or to a video file
        """
        self.path = Path(path) if isinstance(path, str) else path
        self.image_wh = image_wh
        self.crop_to_square = crop_to_square

        if not self.path.exists():
            raise FileNotFoundError(f"The path {self.path} does not exist")

        self.is_video = self.path.is_file() and self.path.suffix.lower() in self.VIDEO_EXTENSIONS

        if not self.is_video:
            raise ValueError(f"The path must be a video file!")

    def get_frames(self, max_frames: Optional[int] = None) -> Generator['np.ndarray', None, None]:
        """
        Generator that yields tuples of (frame_number, frame) from a video file.

        Args:
            max_frames: Maximum number of frames to read (None for all frames)

        Returns:
            Generator yielding tuples of (int, numpy.ndarray)
        """
        if not self.is_video:
            raise ValueError("Cannot get frames from a non-video path")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file {self.path}")

        frame_count = 0

        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame = process_frame(frame, image_wh=self.image_wh, crop_to_square=self.crop_to_square)

                yield frame
                frame_count += 1
        finally:
            cap.release()

class ImageProcessor:
    # Common image file extensions
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    def __init__(self, path: Union[str, Path], image_wh: tuple[int, int], crop_to_square: bool):
        """
        Initialize the MediaProcessor with a path to a directory or a video file.

        Args:
            path: Path to a directory containing images or to a video file
        """
        self.path = Path(path) if isinstance(path, str) else path
        self.image_wh = image_wh
        self.crop_to_square = crop_to_square

        if not self.path.exists():
            raise FileNotFoundError(f"The path {self.path} does not exist")

        self.is_directory = self.path.is_dir()

        if not self.is_directory:
            raise ValueError(f"The path must be either a directory")

    def get_frames(self) -> Generator['np.ndarray', None, None]:
        """
        Generator that yields tuples of (file_path, image) for all images in the directory.

        Returns:
            Generator yielding tuples of (Path, numpy.ndarray)
        """
        if not self.is_directory:
            raise ValueError("Cannot get images from a non-directory path")

        for file_path in self.path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                try:
                    frame = cv2.imread(str(file_path))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame = process_frame(frame, image_wh=self.image_wh, crop_to_square=self.crop_to_square)

                    if frame is not None:
                        yield frame
                except Exception as e:
                    print(f"Error reading image {file_path}: {e}")