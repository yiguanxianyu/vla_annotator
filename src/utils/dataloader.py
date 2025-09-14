import json
from pathlib import Path
from typing import List

import av
import numpy as np
# from transformers import Qwen2.5


"""
Data loader for AgiBotSFT dataset.

This module provides a PyTorch Dataset implementation for loading and processing
robot task execution data for supervised fine-tuning (SFT). The dataset consists of:
- Video observations from robot executions (stored as MP4 files)
- Task information including episode IDs, task names, initial scene descriptions
- Action sequences with frame indices, action text descriptions, and skill types

The dataset follows a specific directory structure:
- data_path/
  - task_info/
    - task_*.json
  - observations/
    - [task_id]/
      - [episode_id]/
        - video/
          - head_color.mp4

Each JSON file contains task information in the format shown in the example below.
"""

"""
[
    {
        "episode_id": 648649,
        "label_info": {
            "action_config": [
                {
                    "start_frame": 8,
                    "end_frame": 218,
                    "action_text": "Retrieve cucumber from the shelf.",
                    "skill": "Pick"
                },
                {
                    "start_frame": 218,
                    "end_frame": 436,
                    "action_text": "Place the held cucumber into the plastic bag in the shopping cart.",
                    "skill": "Place"
                }
            ]
        },
        "task_name": "Pickup items in the supermarket",
        "init_scene_text": "The robot is positioned in front of the fruit stand in the supermarket environment."
    }
]
"""

av.logging.set_level(av.logging.VERBOSE)


class AgiBotSFTDataSet:
    """
    PyTorch Dataset for loading robot task execution data.

    This dataset handles the loading of video observations and corresponding task information
    for robot executions, providing access to episode details, task descriptions,
    and action configurations.
    """

    def __init__(self, data_path: Path, fps: int):
        self.fps = fps
        self.base_path = data_path
        self.task_info_root = data_path / "task_info"
        self.video_root = data_path / "observations"
        self.data_path = []
        self.load_all_data_info()

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        video_path, episode = self.data_path[idx]
        video_frames = self.load_video(video_path, self.fps)
        episode_id = episode["episode_id"]
        task_name = episode["task_name"]
        init_scene_text = episode["init_scene_text"]
        action_config = episode["label_info"]["action_config"]
        return video_frames, episode

    def load_all_data_info(self):
        """
        Load information for all available episodes from JSON files.

        Searches for task_*.json files in the task_info directory and checks if
        corresponding video files exist in the observations directory.
        """
        for jsonfile in self.task_info_root.rglob("task_*.json"):
            with open(jsonfile, "r") as f:
                task_info = json.load(f)

            for episode in task_info:
                video_path = (
                    self.video_root / jsonfile.stem[5:] / str(episode["episode_id"]) / "videos" / "head_color.mp4"
                )
                if video_path.exists():
                    self.data_path.append((video_path, episode))

    def load_video(self, video_path: Path, num_frames: int) -> List[np.ndarray]:
        """
        Evenly sample frames from a video file.

        Args:
            video_path (Path): Path to the video file
            num_frames (int): Number of frames to sample

        Returns:
            list: List of sampled video frames
        """
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]

            assert len(container.streams.video) > 0, f"No video stream found in {video_path}"
            assert video_stream.frames >= num_frames, (
                f"Video has only {video_stream.frames} frames, but {num_frames} are requested."
            )

            frames_to_extract = num_frames
            video_seconds = float(video_stream.time_base * video_stream.duration)
            sample_interval = av.time_base * video_seconds / (frames_to_extract - 1)

            frames = []
            for index in range(frames_to_extract):
                timestamp_us = int(index * sample_interval)
                container.seek(timestamp_us)
                frame = container.decode(video=0)
                image = next(iter(frame)).to_rgb().to_ndarray()
                frames.append(image)

        return frames  # T, H, W, C


if __name__ == "__main__":
    from pprint import pprint

    dataset = AgiBotSFTDataSet(Path("/mnt/nvme/sft_data"), 32)
    print(len(dataset))
    video_frames, episode = dataset[0]
    pprint(episode, indent=2, width=160)
