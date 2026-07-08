"""Capture labeled verification frames from both robot cameras.
Shows camera role, index, resolution, and timestamp overlay.
Also checks: brightness, sync timing, and frame quality.
"""
import cv2
import time
import os
import numpy as np
from datetime import datetime

SAVE_DIR = "D:/sO-101/camera_verification"
os.makedirs(SAVE_DIR, exist_ok=True)

# This must match record_episode.py cam_map exactly
CAM_MAP = {"cam_high": 1, "cam_wrist": 2}

results = {}

for name, idx in CAM_MAP.items():
    print(f"\n--- Opening {name} (index {idx}) ---")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  FAILED to open index {idx}!")
        continue

    # Warmup: let auto-exposure settle
    print(f"  Warming up (60 frames)...", end="", flush=True)
    for _ in range(60):
        ret, frame = cap.read()
        time.sleep(0.05)
    print(" done.")

    # Capture final frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  ERROR: No frame after warmup!")
        cap.release()
        continue

    h, w = frame.shape[:2]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Analyze frame quality
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    is_black = mean_brightness < 15
    is_overexposed = mean_brightness > 240

    results[name] = {
        "index": idx,
        "resolution": f"{w}x{h}",
        "brightness_mean": round(mean_brightness, 1),
        "brightness_std": round(std_brightness, 1),
        "is_black": is_black,
        "is_overexposed": is_overexposed,
        "capture_time": ts,
    }

    # --- Draw label overlay on frame ---
    label_role = "OVERHEAD CAMERA" if name == "cam_high" else "WRIST CAMERA"
    color = (0, 255, 0)  # green

    # Semi-transparent black banner at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Text overlays
    cv2.putText(frame, f"{label_role}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Index: {idx} | {w}x{h} | {name}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Brightness: {mean_brightness:.0f} (std={std_brightness:.0f}) | {ts}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Quality warnings
    if is_black:
        cv2.putText(frame, "WARNING: BLACK FRAME!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if is_overexposed:
        cv2.putText(frame, "WARNING: OVEREXPOSED!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Save
    path = os.path.join(SAVE_DIR, f"VERIFIED_{name}_idx{idx}.jpg")
    cv2.imwrite(path, frame)
    print(f"  Saved: {path}")
    print(f"  Resolution: {w}x{h}")
    print(f"  Brightness: mean={mean_brightness:.1f}, std={std_brightness:.1f}")

    cap.release()

# --- Sync timing check ---
print("\n\n" + "=" * 60)
print("  SIMULTANEOUS CAPTURE TEST (sync check)")
print("=" * 60)

caps = {}
for name, idx in CAM_MAP.items():
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        # Quick warmup
        for _ in range(10):
            cap.read()
        caps[name] = cap

if len(caps) == 2:
    # Capture both as fast as possible
    t1 = time.perf_counter()
    ret1, f1 = caps["cam_high"].read()
    t_mid = time.perf_counter()
    ret2, f2 = caps["cam_wrist"].read()
    t2 = time.perf_counter()

    delta_ms = (t2 - t1) * 1000
    print(f"  Time between captures: {delta_ms:.1f} ms")
    print(f"  cam_high capture: {(t_mid - t1)*1000:.1f} ms")
    print(f"  cam_wrist capture: {(t2 - t_mid)*1000:.1f} ms")

    if delta_ms < 100:
        print("  ✓ Cameras are well-synced (< 100ms apart)")
    elif delta_ms < 200:
        print("  ⚠ Cameras are loosely synced (100-200ms apart)")
    else:
        print("  ✗ WARNING: Cameras are NOT synced (> 200ms apart)")

    results["sync_delta_ms"] = round(delta_ms, 1)
else:
    print("  ERROR: Could not open both cameras for sync test!")

for cap in caps.values():
    cap.release()

# --- Summary ---
print("\n\n" + "=" * 60)
print("  VERIFICATION SUMMARY")
print("=" * 60)
for name, info in results.items():
    if isinstance(info, dict):
        status = "✓ OK" if not info.get("is_black") and not info.get("is_overexposed") else "✗ PROBLEM"
        print(f"  {name} (index {info['index']}): {info['resolution']} | brightness={info['brightness_mean']} | {status}")

print(f"\n  Labeled images saved to: {SAVE_DIR}")
print("  Done.")
