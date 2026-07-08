#!/usr/bin/env python
"""Audit all recorded episodes for data integrity."""
import json, os, shutil

base = os.path.expanduser("~/.cache/huggingface/lerobot/siddhantjain/pick_bottle_training")
out_dir = "D:/sO-101/episode_audit_frames"
os.makedirs(out_dir, exist_ok=True)

episodes = sorted([d for d in os.listdir(base) if d.startswith("episode_")])
print(f"Total episodes on disk: {len(episodes)}\n")

header = f"{'Ep':>4} | {'Frames':>6} | {'Actions':>7} | {'States':>6} | {'CamH':>5} | {'CamW':>5} | {'Dur(s)':>6} | {'FPS':>4} | Status"
print(header)
print("-" * len(header))

all_pass = True
total_frames = 0
total_dur = 0.0

for ep_name in episodes:
    ep_dir = os.path.join(base, ep_name)
    with open(os.path.join(ep_dir, "episode_data.json")) as f:
        data = json.load(f)
    
    nf = data.get("num_frames", 0)
    na = len(data.get("actions", []))
    ns = len(data.get("states", []))
    dur = round(data.get("duration_s", 0), 1)
    fps = data.get("fps", 0)
    
    cam_h_dir = os.path.join(ep_dir, "frames_cam_high")
    cam_w_dir = os.path.join(ep_dir, "frames_cam_wrist")
    ch = len([f for f in os.listdir(cam_h_dir) if f.endswith(".jpg")])
    cw = len([f for f in os.listdir(cam_w_dir) if f.endswith(".jpg")])
    
    ok = (nf == na == ns == ch == cw)
    if not ok:
        all_pass = False
    
    num = ep_name.replace("episode_", "")
    status = "PASS" if ok else "FAIL"
    print(f"{num:>4} | {nf:>6} | {na:>7} | {ns:>6} | {ch:>5} | {cw:>5} | {dur:>6} | {fps:>4} | {status}")
    
    total_frames += nf
    total_dur += dur
    
    # Copy first and last frame from each camera for visual check
    for cam in ["cam_high", "cam_wrist"]:
        frames_dir = os.path.join(ep_dir, f"frames_{cam}")
        for label, idx in [("first", 0), ("last", nf - 1)]:
            src = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
            dst = os.path.join(out_dir, f"ep{num}_{cam}_{label}.jpg")
            if os.path.exists(src):
                shutil.copy2(src, dst)

print()
print(f"TOTAL: {total_frames} frames across {len(episodes)} episodes ({total_dur:.0f}s)")
if all_pass:
    print("OVERALL: ALL PASS")
else:
    print("OVERALL: SOME FAILED")
print(f"\nSample frames saved to: {out_dir}")
