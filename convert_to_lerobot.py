#!/usr/bin/env python
"""
Convert raw SO-100 dataset (JPG + JSON) → LeRobot v2 (Parquet + MP4).

Usage:
    python convert_to_lerobot.py \
        --raw-dir  "C:/Users/asus vivoBook/.cache/huggingface/lerobot/siddhantjain/pick_bottle_training" \
        --repo-id  "Raakshass/so100_pick_bottle" \
        --fps 15

What this script does:
    1. Reads each episode_XXXX directory (frames_cam_high, frames_cam_wrist, episode_data.json)
    2. Creates a LeRobotDataset with video features for both cameras
    3. For each frame, calls dataset.add_frame() with PIL images, numpy state/action, and task string
    4. Calls dataset.save_episode() after each episode
    5. Calls dataset.finalize() to flush all metadata

Output directory: $HF_LEROBOT_HOME/{repo_id}  (or --output-dir if specified)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure lerobot is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ── Joint ordering ──────────────────────────────────────────────────
# The JSON stores named joints. We define a canonical ordering for the
# 6-DOF SO-100 arm that matches what policies expect.
JOINT_ORDER = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def json_dict_to_array(d: dict, keys: list[str]) -> np.ndarray:
    """Convert a JSON object with named joints to a float32 numpy array."""
    return np.array([d[k] for k in keys], dtype=np.float32)


def get_episode_dirs(raw_dir: Path) -> list[Path]:
    """Return sorted list of episode_XXXX directories."""
    dirs = sorted(raw_dir.glob("episode_*"))
    if not dirs:
        raise FileNotFoundError(f"No episode_* directories found in {raw_dir}")
    return dirs


def build_features() -> dict:
    """
    Define the LeRobot feature specification for our dataset.

    This tells LeRobot:
      - observation.images.cam_high → video, shape (3, 480, 640)
      - observation.images.cam_wrist → video, shape (3, 480, 640)
      - observation.state → float32, shape (6,)
      - action → float32, shape (6,)
    """
    return {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "motors": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "motors": [
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ]
            },
        },
    }


def convert(raw_dir: Path, repo_id: str, fps: int, output_dir: Path | None = None):
    """Main conversion pipeline."""

    episode_dirs = get_episode_dirs(raw_dir)
    num_episodes = len(episode_dirs)
    features = build_features()

    print("=" * 56)
    print("  SO-100 -> LeRobot v2 Converter")
    print("=" * 56)
    print(f"  Raw directory : {str(raw_dir)}")
    print(f"  Repo ID       : {repo_id}")
    print(f"  FPS           : {fps}")
    print(f"  Episodes      : {num_episodes}")
    print("=" * 56)
    print()

    # ── Create the dataset ──────────────────────────────────────────
    root = str(output_dir) if output_dir else None
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type="so100",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )

    total_frames = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        ep_name = ep_dir.name

        # ── Load episode JSON ───────────────────────────────────────
        json_path = ep_dir / "episode_data.json"
        with open(json_path, "r") as f:
            ep_data = json.load(f)

        task_str = ep_data["task"]
        num_frames = ep_data["num_frames"]
        actions = ep_data["actions"]
        states = ep_data["states"]

        cam_high_dir = ep_dir / "frames_cam_high"
        cam_wrist_dir = ep_dir / "frames_cam_wrist"

        # Sanity checks
        assert len(actions) == num_frames, f"{ep_name}: actions count mismatch"
        assert len(states) == num_frames, f"{ep_name}: states count mismatch"

        cam_high_files = sorted(cam_high_dir.glob("frame_*.jpg"))
        cam_wrist_files = sorted(cam_wrist_dir.glob("frame_*.jpg"))
        assert len(cam_high_files) == num_frames, f"{ep_name}: cam_high frame count mismatch"
        assert len(cam_wrist_files) == num_frames, f"{ep_name}: cam_wrist frame count mismatch"

        # ── Add each frame ──────────────────────────────────────────
        pbar = tqdm(range(num_frames), desc=f"Episode {ep_idx:03d}/{num_episodes-1:03d}", leave=False)
        for frame_idx in pbar:
            # Load images as PIL (LeRobot accepts PIL for video features)
            img_high = Image.open(cam_high_files[frame_idx])
            img_wrist = Image.open(cam_wrist_files[frame_idx])

            # Convert joint dicts to numpy arrays
            state_arr = json_dict_to_array(states[frame_idx], JOINT_ORDER)
            action_arr = json_dict_to_array(actions[frame_idx], JOINT_ORDER)

            frame = {
                "observation.images.cam_high": img_high,
                "observation.images.cam_wrist": img_wrist,
                "observation.state": state_arr,
                "action": action_arr,
                "task": task_str,
            }
            dataset.add_frame(frame)

        # ── Save the episode ────────────────────────────────────────
        dataset.save_episode()
        total_frames += num_frames
        print(f"  [OK] {ep_name}: {num_frames} frames saved (total: {total_frames})")

    # ── Finalize ────────────────────────────────────────────────────
    print()
    print("Finalizing dataset (flushing metadata + computing stats)...")
    dataset.finalize()

    print()
    print("=" * 56)
    print("  CONVERSION COMPLETE")
    print("=" * 56)
    print(f"  Total episodes : {num_episodes}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Dataset root   : {dataset.root}")
    print("=" * 56)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw SO-100 data to LeRobot v2 format")
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to raw dataset directory containing episode_XXXX folders",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Raakshass/so100_pick_bottle",
        help="HuggingFace repo ID for the dataset (default: Raakshass/so100_pick_bottle)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: $HF_LEROBOT_HOME/{repo_id})",
    )
    args = parser.parse_args()

    convert(
        raw_dir=Path(args.raw_dir),
        repo_id=args.repo_id,
        fps=args.fps,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
