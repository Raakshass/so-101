"""Capture a frame from all 3 detected cameras with proper warmup."""
import cv2
import time
import os

save_dir = "D:/sO-101/camera_probes"
os.makedirs(save_dir, exist_ok=True)

for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Index {idx}: NOT AVAILABLE")
        continue

    # Warmup: read several frames to let auto-exposure settle
    for _ in range(60):
        ret, frame = cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    if ret and frame is not None:
        path = os.path.join(save_dir, f"camera_index_{idx}.jpg")
        cv2.imwrite(path, frame)
        h, w = frame.shape[:2]
        print(f"Index {idx}: {w}x{h} -> Saved {path}")
    else:
        print(f"Index {idx}: OPENED but no frame after warmup")

    cap.release()

print("Done.")
