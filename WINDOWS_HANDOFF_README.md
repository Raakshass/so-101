# LeRobot SO-100 Windows Handoff

Welcome to the Windows transition for the SO-100 data collection project! 

This repository contains custom scripts written to interface with two SO-100 robot arms (Leader and Follower) and two USB cameras (Overhead and Wrist) using the LeRobot framework.

## Project Context
We are attempting to record a 20-episode dataset for training a Vision-Language-Action (VLA) model to pick up a bottle. We faced severe issues on macOS due to USB camera indices randomly reshuffling whenever the USB hub was reset, which resulted in accidentally recording 10 episodes using the Mac's built-in webcam instead of the robot's wrist camera.

To avoid this on Windows, **we must strictly verify the camera indices before recording any data.**

## Hardware Setup
1. **Robot Arms (SO-100)**: 
   - 1 Leader arm (controlling)
   - 1 Follower arm (acting)
2. **Cameras**:
   - 1 Overhead USB Camera (capturing the workspace)
   - 1 Wrist USB Camera (mounted on the Follower arm)

## Custom Scripts Included
1. `scan_motors.py`: Scans and identifies the serial ports of the Dynamixel motors for the Follower arm.
2. `scan_leader.py`: Scans and identifies the serial port of the Dynamixel motors for the Leader arm.
3. `record_episode.py`: The main data collection script. It connects to the cameras, initializes the leader and follower arms, and provides a CLI (`start`, `stop`, `quit`) to record demonstrations. It saves episodes directly to `~/.cache/huggingface/lerobot/siddhantjain/pick_bottle_training/`.

## Setup Instructions for Windows
1. **Install Anaconda/Miniconda** if you haven't already.
2. **Clone this repository** to your Windows machine:
   ```bash
   git clone https://github.com/Raakshass/so-101.git
   cd so-101
   ```
3. **Create the Conda Environment**:
   ```bash
   conda create -y -n lerobot python=3.10
   conda activate lerobot
   ```
4. **Install LeRobot and Dependencies**:
   Follow the standard LeRobot Windows installation instructions. You will likely need to install PyTorch with CUDA support if you have an NVIDIA GPU, and then install LeRobot dependencies.
5. **Verify Serial Ports**:
   Run `scan_motors.py` and `scan_leader.py` to find the `COM` ports for your leader and follower arms on Windows (they won't be `/dev/tty.usbmodem...` anymore). Update `record_episode.py` lines 122 and 147 with the new `COM` ports.
6. **Verify Camera Indices (CRITICAL)**:
   Windows handles USB camera indices differently than macOS. Write a short script using `cv2` to probe all camera indices (0, 1, 2, 3) and visually confirm which index corresponds to the Overhead camera and which corresponds to the Wrist camera. Update `record_episode.py` line 95 with the correct indices!

Once verified, run `python record_episode.py` to start recording!
