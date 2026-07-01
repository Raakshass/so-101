#!/usr/bin/env python
"""
Simple pick-and-place recording script.
Control via stdin: type 'start' to begin recording, 'stop' to end.
Uses raw cv2.VideoCapture with background threads for high FPS on macOS.
"""
import sys
import os
import time
import json
import threading
import cv2
import numpy as np

from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig


class ThreadedCamera:
    """Reads frames from a cv2.VideoCapture in a background thread."""
    def __init__(self, index):
        self.index = index
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        for attempt in range(5):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Verify we can actually read a frame
                for _ in range(30):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.frame = frame
                        return
                    time.sleep(0.1)
                # Opened but no frames, release and try again
                cap.release()
            time.sleep(1.0)
            print(f"  Camera {index}: Retrying connection ({attempt+1}/5)...", flush=True)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        """Start the background reading thread."""
        if not self.is_opened():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Return the latest frame from the background buffer."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()


def main():
    follower_port = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem5A7C1208251"
    leader_port = sys.argv[2] if len(sys.argv) > 2 else "/dev/tty.usbmodem5AAF2883381"
    save_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser(
        "~/.cache/huggingface/lerobot/siddhantjain/pick_bottle_training"
    )

    os.makedirs(save_dir, exist_ok=True)

    # Figure out next episode number
    existing = [d for d in os.listdir(save_dir) if d.startswith("episode_")]
    episode_num = len(existing)

    # --- Connect cameras first ---
    print("Connecting cameras...")
    cams = []
    cam_map = {"cam_wrist": 1, "cam_high": 0}
    for name, idx in cam_map.items():
        print(f"  Opening {name} (index {idx})...", flush=True)
        tcam = ThreadedCamera(idx)
        if tcam.is_opened():
            if tcam.start():
                cams.append((name, tcam))
                print(f"  Connected {name} -> camera index {idx}", flush=True)
            else:
                print(f"  WARNING: {name} (index {idx}) opened but no frames!", flush=True)
                tcam.release()
        else:
            print(f"  WARNING: {name} (index {idx}) failed to open!", flush=True)

    if not cams:
        print("  FATAL: No cameras found! Exiting.")
        return

    # --- Connect robot arms ---
    print("Connecting follower arm...")
    follower_cfg = SOFollowerRobotConfig(port=follower_port, id="my_follower")
    follower = SOFollower(follower_cfg)
    follower.connect()
    print("  Follower connected.")

    print("Connecting leader arm...")
    leader_cfg = SOLeaderTeleopConfig(port=leader_port, id="my_leader")
    leader = SOLeader(leader_cfg)
    leader.connect()
    print("  Leader connected.")

    print()
    print("=" * 60)
    print(f"  READY TO RECORD EPISODE {episode_num}")
    print("  Type 'start' and press Enter to begin recording.")
    print("  Type 'stop' and press Enter to stop and save.")
    print("  Type 'quit' to exit without recording.")
    print("=" * 60)
    print()
    sys.stdout.flush()

    # --- Wait for 'start' command ---
    while True:
        cmd = input(">> ").strip().lower()
        if cmd == "start":
            break
        elif cmd == "quit":
            print("Quitting without recording.")
            follower.disconnect()
            leader.disconnect()
            for _, c in cams:
                c.release()
            return
        else:
            print(f"Unknown command '{cmd}'. Type 'start' or 'quit'.")
            sys.stdout.flush()

    # --- Recording loop ---
    ep_dir = os.path.join(save_dir, f"episode_{episode_num:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    frames_dirs = {}
    for name, _ in cams:
        d = os.path.join(ep_dir, f"frames_{name}")
        os.makedirs(d, exist_ok=True)
        frames_dirs[name] = d

    actions = []
    states = []
    timestamps = []
    frame_count = 0
    fps_target = 15
    frame_interval = 1.0 / fps_target

    # Use a thread to listen for 'stop'
    stop_event = threading.Event()

    def listen_for_stop():
        while not stop_event.is_set():
            try:
                line = input()
                if line.strip().lower() == "stop":
                    stop_event.set()
                    break
            except EOFError:
                break

    stop_thread = threading.Thread(target=listen_for_stop, daemon=True)
    stop_thread.start()

    print(f"\n  RECORDING EPISODE {episode_num}...")
    print("  Teleoperate now! Move the Leader arm to control the Follower.")
    print("  Type 'stop' and press Enter when done.\n")
    sys.stdout.flush()

    start_time = time.time()
    last_print_time = start_time

    while not stop_event.is_set():
        loop_start = time.perf_counter()

        # Read leader action and send to follower
        action = leader.get_action()
        follower.send_action(action)

        # Read follower state
        state = follower.get_observation()

        # Read camera frames (non-blocking from background threads)
        frames = {}
        for name, tcam in cams:
            frame = tcam.read()
            if frame is not None:
                frames[name] = frame

        # Store data
        timestamp = time.time() - start_time
        actions.append(action)
        states.append(state)
        timestamps.append(timestamp)

        # Save frames
        for name, frame in frames.items():
            frame_path = os.path.join(frames_dirs[name], f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)

        frame_count += 1

        # Print status every 2 seconds
        now = time.time()
        if now - last_print_time >= 2.0:
            elapsed = now - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"  [{elapsed:.0f}s] frames={frame_count} | fps={actual_fps:.1f}")
            sys.stdout.flush()
            last_print_time = now

        # Maintain target FPS
        elapsed_frame = time.perf_counter() - loop_start
        sleep_time = frame_interval - elapsed_frame
        if sleep_time > 0:
            time.sleep(sleep_time)

    # --- Save episode data ---
    elapsed = time.time() - start_time
    print(f"\n  Recording stopped. {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print("  Saving episode data...")
    sys.stdout.flush()

    episode_data = {
        "episode_num": episode_num,
        "task": "pick up bottle and place it in a yellow square",
        "num_frames": frame_count,
        "duration_s": elapsed,
        "fps": fps_target,
        "actions": actions,
        "states": states,
        "timestamps": timestamps,
    }

    with open(os.path.join(ep_dir, "episode_data.json"), "w") as f:
        json.dump(episode_data, f, indent=2, default=str)

    print(f"  Episode {episode_num} saved to: {ep_dir}")
    for name, d in frames_dirs.items():
        print(f"  Frames ({name}): {d}")
    print(f"  Data: {os.path.join(ep_dir, 'episode_data.json')}")
    sys.stdout.flush()

    # Cleanup
    for _, c in cams:
        c.release()
    follower.disconnect()
    leader.disconnect()
    print("\n  All disconnected. Done!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
