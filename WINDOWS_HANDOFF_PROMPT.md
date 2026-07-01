# Prompt for Windows AI Agent

*Copy and paste the following text block to the AI agent on your Windows laptop:*

***

Hello! We are transitioning a robotics data collection project from a macOS machine to this Windows machine. I need your help to get everything set up and record 20 episodes from scratch.

### Context
I have a custom robotic setup using the Hugging Face LeRobot framework. The hardware consists of:
1.  **Leader Arm**: An SO-100 robotic arm (acting as the teleoperation controller).
2.  **Follower Arm**: An SO-100 robotic arm (acting as the robot being controlled).
3.  **Overhead Camera**: A USB webcam capturing the overall workspace (a white mat).
4.  **Wrist Camera**: A USB webcam mounted on the Follower arm's gripper.

### What We Have
I have cloned my GitHub repository (`https://github.com/Raakshass/so-101.git`) which contains three custom scripts that we perfected on the previous machine:
1.  `scan_motors.py`: A script to find the serial port of the Follower arm.
2.  `scan_leader.py`: A script to find the serial port of the Leader arm.
3.  `record_episode.py`: The main script that connects to both cameras, connects to both arms, and provides a CLI (`start`, `stop`, `quit`) to record episodes.

### Your Tasks
Please guide me step-by-step through the following process. Do not rush, let's do this methodically:

**Step 1: Environment Setup**
Ensure I have a Conda environment called `lerobot` with Python 3.10 and all the correct dependencies installed for Windows (including PyTorch with CUDA if applicable).

**Step 2: Serial Port Configuration**
The custom scripts `record_episode.py` currently hardcode macOS serial ports (e.g., `/dev/tty.usbmodem...`). 
Please run `scan_motors.py` and `scan_leader.py` to identify the correct `COM` ports for the Leader and Follower arms on this Windows machine. Once identified, update `record_episode.py` lines 122 and 147 with the new `COM` ports.

**Step 3: Camera Index Verification (CRITICAL)**
We had a massive failure on the previous machine where macOS reshuffled the camera indices and we accidentally recorded 10 episodes using the laptop's built-in webcam instead of the robot's wrist camera. 
*Do not assume the indices in `record_episode.py` are correct!* 
Please write a short Python script using OpenCV to capture a single frame from every available camera index (0, 1, 2, 3...) and save them to disk. I will look at the frames and tell you which index is the Overhead camera and which is the Wrist camera. Only after I confirm, update `cam_map` on line 95 of `record_episode.py`.

**Step 4: Record Episodes**
Once everything is verified, launch `record_episode.py` and let's start recording 20 episodes from scratch! The goal is to record demonstrations of the robot picking up a bottle.

Let's begin with Step 1. Are you ready?
