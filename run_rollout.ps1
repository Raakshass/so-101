# =====================================================================
#  SO-100 Autonomous Rollout — SmolVLA "pick up bottle, place in yellow square"
#  Launches lerobot-rollout with RTC inference on the physical follower arm.
#
#  Verified 2026-07-04:
#    - lerobot 0.5.2, torch 2.11+cu126, CUDA on RTX 3050 Laptop GPU
#    - Cameras (DirectShow): index 1 = cam_high (overhead), index 2 = cam_wrist
#    - Follower on COM12, calibration id = my_follower (cached)
#    - Model Raakshass/smolvla_so100_pick_bottle expects cam_high, cam_wrist, state[6]
#
#  Corrections vs. the original command:
#    + --robot.id=my_follower   (load cached calibration; default id=None would re-calibrate)
#    + --fps=15                 (match 15fps training data; rollout default is 30 = 2x too fast)
#    + backend: DSHOW           (reliable Windows capture; lerobot default ANY->MSMF was flaky)
#    + execution_horizon=30     (was 10; inference latency is ~18-21 ticks on the RTX 3050,
#                                so a 10-tick horizon starved the RTC queue -> arm vibrated.
#                                30 ticks (2.0s) > inference latency -> smooth motion.)
#
#  One-time deps installed:  pip install num2words   (required by the SmolVLM processor)
# =====================================================================

$ErrorActionPreference = "Stop"
$envRoot = "$env:USERPROFILE\miniconda3\envs\lerobot"
$py = "$envRoot\python.exe"

# Put the env's Scripts + Library\bin on PATH so the Rerun viewer (rerun.exe) and
# DLLs resolve when we call python.exe directly without `conda activate`.
$env:PATH = "$envRoot;$envRoot\Library\bin;$envRoot\Scripts;$env:PATH"

# Prevent cp1252 UnicodeEncodeError on Windows console when logs contain e.g. the >= sign.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# --- Editable knobs (change here if hardware moves) ---
$FollowerPort = "COM12"
$RobotId      = "my_follower"
$HighCamIdx   = 1     # overhead (cam_high)
$WristCamIdx  = 2     # wrist    (cam_wrist)
$Task         = "pick up bottle and place it in a yellow square"
$Duration     = 60    # seconds, then auto-stop
$Fps          = 15
$ExecHorizon  = 30    # RTC actions committed per chunk; keep > inference latency (~20 ticks)

$cameras = "{ cam_high: {type: opencv, index_or_path: $HighCamIdx, width: 640, height: 480, fps: 30, backend: DSHOW}, cam_wrist: {type: opencv, index_or_path: $WristCamIdx, width: 640, height: 480, fps: 30, backend: DSHOW}}"

Write-Host "Launching autonomous rollout — keep a hand on Ctrl+C to stop early." -ForegroundColor Yellow

& $py -m lerobot.scripts.lerobot_rollout `
    --strategy.type=base `
    --policy.path=Raakshass/smolvla_so100_pick_bottle `
    --inference.type=rtc `
    --inference.rtc.execution_horizon=$ExecHorizon `
    --robot.type=so100_follower `
    --robot.port=$FollowerPort `
    --robot.id=$RobotId `
    --robot.cameras="$cameras" `
    --task="$Task" `
    --fps=$Fps `
    --duration=$Duration `
    --display_data=true
