"""
Camera Index Probe — CRITICAL pre-recording verification.

Probes camera indices 0-9 using DirectShow backend (Windows),
captures a single frame from each available camera, and saves to disk.

YOU MUST visually inspect the saved images before updating record_episode.py.
"""
import os
import sys
import cv2


def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_probes")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  CAMERA INDEX PROBE")
    print("  Probing indices 0-9 with DirectShow backend...")
    print("=" * 60)
    print()

    found = []

    for idx in range(10):
        # Use DirectShow backend on Windows for reliability
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print(f"  Index {idx}: NOT AVAILABLE")
            cap.release()
            continue

        # Try to read a frame (with retries for slow cameras)
        frame = None
        for attempt in range(15):
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
                break

        if frame is None:
            print(f"  Index {idx}: OPENED but no frames received")
            cap.release()
            continue

        h, w = frame.shape[:2]
        backend = cap.getBackendName()
        fps = cap.get(cv2.CAP_PROP_FPS)

        fname = f"camera_index_{idx}.jpg"
        fpath = os.path.join(save_dir, fname)
        cv2.imwrite(fpath, frame)

        print(f"  Index {idx}: {w}x{h} @ {fps:.0f}fps [{backend}] -> Saved to {fname}")
        found.append(idx)
        cap.release()

    print()
    print("=" * 60)
    print(f"  FOUND {len(found)} cameras at indices: {found}")
    print(f"  Frames saved to: {save_dir}")
    print()
    print("  >> INSPECT THE IMAGES VISUALLY <<")
    print("  Tell me which index is the OVERHEAD camera")
    print("  and which index is the WRIST camera.")
    print("=" * 60)


if __name__ == "__main__":
    main()
