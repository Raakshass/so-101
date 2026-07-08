"""
Deep Data Quality Audit v2 for SO-101 Pick Bottle Training Dataset
===================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed")
    sys.exit(1)

DATASET_DIR = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "lerobot" / "siddhantjain" / "pick_bottle_training"
JOINT_NAMES = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
SAMPLE_RATE = 5


def check_image_quality(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return {"corrupt": True}
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "corrupt": False, "resolution": (w, h), "channels": c,
        "brightness": brightness, "contrast": contrast,
        "blur_score": laplacian_var,
        "is_black": brightness < 10, "is_white": brightness > 245,
    }


def actions_to_array(actions_list):
    """Convert list of joint-name dicts to numpy array."""
    rows = []
    for a in actions_list:
        rows.append([a[j] for j in JOINT_NAMES])
    return np.array(rows, dtype=np.float32)


def main():
    print("=" * 70)
    print("  DEEP DATA QUALITY AUDIT v2")
    print("  Dataset: %s" % DATASET_DIR)
    print("=" * 70)

    if not DATASET_DIR.exists():
        print("ERROR: Dataset directory not found")
        sys.exit(1)

    episode_dirs = sorted(DATASET_DIR.glob("episode_*"))
    n_episodes = len(episode_dirs)

    # =====================================================================
    # [1] STRUCTURAL INTEGRITY
    # =====================================================================
    print("\n[1/6] STRUCTURAL INTEGRITY")
    print("  Episodes found: %d" % n_episodes)
    ep_numbers = [int(d.name.split("_")[1]) for d in episode_dirs]
    expected = list(range(min(ep_numbers), max(ep_numbers) + 1))
    missing = set(expected) - set(ep_numbers)
    if missing:
        print("  WARNING: Missing episodes: %s" % sorted(missing))
    else:
        print("  Episode numbering: CONTIGUOUS [%d-%d]" % (min(ep_numbers), max(ep_numbers)))

    # =====================================================================
    # [2] IMAGE QUALITY
    # =====================================================================
    print("\n[2/6] IMAGE QUALITY ANALYSIS (sampling every %dth frame)" % SAMPLE_RATE)
    print("  %4s | %5s | %5s | %7s | %5s | %5s | %7s | %s" % ("Ep", "CamH", "CamW", "Corrupt", "Black", "White", "Blur<50", "Status"))
    print("  " + "-" * 65)

    total_corrupt = total_black = total_white = total_blurry = total_checked = 0
    all_issues = []

    for ep_dir in episode_dirs:
        ep_num = int(ep_dir.name.split("_")[1])
        ep_issues = []
        cam_high_frames = sorted((ep_dir / "frames_cam_high").glob("*.jpg"))
        cam_wrist_frames = sorted((ep_dir / "frames_cam_wrist").glob("*.jpg"))
        n_high, n_wrist = len(cam_high_frames), len(cam_wrist_frames)

        if n_high != n_wrist:
            ep_issues.append("CAMERA DESYNC: cam_high=%d, cam_wrist=%d" % (n_high, n_wrist))

        ep_corrupt = ep_black = ep_white = ep_blurry = 0
        for cam_name, frames in [("cam_high", cam_high_frames), ("cam_wrist", cam_wrist_frames)]:
            for i in range(0, len(frames), SAMPLE_RATE):
                total_checked += 1
                q = check_image_quality(frames[i])
                if q["corrupt"]:
                    ep_corrupt += 1; total_corrupt += 1
                    ep_issues.append("CORRUPT: %s frame %d" % (cam_name, i))
                else:
                    if q["is_black"]:
                        ep_black += 1; total_black += 1
                        ep_issues.append("BLACK FRAME: %s frame %d" % (cam_name, i))
                    if q["is_white"]:
                        ep_white += 1; total_white += 1
                    if q["blur_score"] < 50:
                        ep_blurry += 1; total_blurry += 1
                    if q["resolution"] != (640, 480):
                        ep_issues.append("RESOLUTION: %s frame %d = %s" % (cam_name, i, q["resolution"]))

        status = "PASS" if not ep_issues else ("FAIL" if ep_corrupt > 0 else "WARN")
        print("  %04d | %5d | %5d | %7d | %5d | %5d | %7d | %s" % (ep_num, n_high, n_wrist, ep_corrupt, ep_black, ep_white, ep_blurry, status))
        if ep_issues:
            all_issues.extend([(ep_num, issue) for issue in ep_issues])

    print("\n  TOTALS: %d frames checked" % total_checked)
    print("    Corrupt: %d  |  Black: %d  |  White: %d  |  Blurry(<50): %d" % (total_corrupt, total_black, total_white, total_blurry))

    # =====================================================================
    # [3] CAMERA SYNC
    # =====================================================================
    print("\n[3/6] CAMERA SYNC VERIFICATION")
    sync_issues = 0
    for ep_dir in episode_dirs:
        ep_num = int(ep_dir.name.split("_")[1])
        ch = sorted((ep_dir / "frames_cam_high").glob("*.jpg"))
        cw = sorted((ep_dir / "frames_cam_wrist").glob("*.jpg"))
        if len(ch) != len(cw):
            print("  Ep %04d: DESYNC cam_high=%d vs cam_wrist=%d" % (ep_num, len(ch), len(cw)))
            sync_issues += 1
        else:
            high_names = [f.stem for f in ch]
            wrist_names = [f.stem for f in cw]
            if high_names != wrist_names:
                mismatches = sum(1 for a, b in zip(high_names, wrist_names) if a != b)
                print("  Ep %04d: FILENAME MISMATCH (%d frames)" % (ep_num, mismatches))
                sync_issues += 1

    if sync_issues == 0:
        print("  ALL %d episodes: cameras perfectly synced (frame counts and indices match)" % n_episodes)

    # =====================================================================
    # [4] ACTION/STATE DISTRIBUTION
    # =====================================================================
    print("\n[4/6] ACTION/STATE DISTRIBUTION ANALYSIS")
    global_actions = []
    global_states = []
    ep_frame_counts = []
    ep_durations = []
    timing_jitters = []
    constant_joints_eps = []

    for ep_dir in episode_dirs:
        ep_num = int(ep_dir.name.split("_")[1])
        json_path = ep_dir / "episode_data.json"
        if not json_path.exists():
            print("  Ep %04d: NO DATA FILE" % ep_num)
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        n_frames = data.get("num_frames", 0)
        ep_frame_counts.append(n_frames)
        ep_durations.append(data.get("duration_s", 0))

        # Actions
        actions_list = data.get("actions", [])
        states_list = data.get("states", [])
        timestamps = data.get("timestamps", [])

        if not actions_list:
            print("  Ep %04d: No actions" % ep_num)
            continue

        actions_np = actions_to_array(actions_list)
        global_actions.append(actions_np)

        if states_list:
            states_np = actions_to_array(states_list)
            global_states.append(states_np)

        # Check for constant joints in this episode
        for j in range(actions_np.shape[1]):
            col = actions_np[:, j]
            if np.std(col) < 0.5:
                constant_joints_eps.append((ep_num, JOINT_NAMES[j], float(np.std(col))))

        # Check sudden jumps per joint
        for j in range(actions_np.shape[1]):
            col = actions_np[:, j]
            diffs = np.abs(np.diff(col))
            if len(diffs) > 0 and np.std(diffs) > 0:
                threshold = np.mean(diffs) + 4 * np.std(diffs)
                n_jumps = int(np.sum(diffs > threshold))
                if n_jumps > 3:
                    all_issues.append((ep_num, "SUDDEN JUMPS: %s has %d jumps" % (JOINT_NAMES[j], n_jumps)))

        # Timing jitter
        if len(timestamps) > 1:
            ts = np.array(timestamps)
            dt = np.diff(ts)
            mean_dt = np.mean(dt)
            std_dt = np.std(dt)
            jitter_pct = (std_dt / mean_dt * 100) if mean_dt > 0 else 0
            timing_jitters.append(jitter_pct)

            # Flag frames with > 2x expected dt (dropped frames)
            expected_dt = 1.0 / data.get("fps", 15)
            dropped = int(np.sum(dt > 2.5 * expected_dt))
            if dropped > 0:
                all_issues.append((ep_num, "DROPPED FRAMES: %d intervals > 2.5x expected dt" % dropped))

    # Global stats
    if global_actions:
        all_act = np.concatenate(global_actions, axis=0)
        n_total = all_act.shape[0]
        print("\n  Global Action Statistics (%d total frames, %d joints):" % (n_total, all_act.shape[1]))
        print("  %20s | %10s | %10s | %10s | %10s | %10s" % ("Joint", "Min", "Max", "Mean", "Std", "Range"))
        print("  " + "-" * 77)
        for j in range(all_act.shape[1]):
            col = all_act[:, j]
            print("  %20s | %10.2f | %10.2f | %10.2f | %10.2f | %10.2f" % (
                JOINT_NAMES[j], np.min(col), np.max(col), np.mean(col), np.std(col), np.ptp(col)))

    if global_states:
        all_st = np.concatenate(global_states, axis=0)
        print("\n  Global State Statistics (%d total frames, %d joints):" % (all_st.shape[0], all_st.shape[1]))
        print("  %20s | %10s | %10s | %10s | %10s | %10s" % ("Joint", "Min", "Max", "Mean", "Std", "Range"))
        print("  " + "-" * 77)
        for j in range(all_st.shape[1]):
            col = all_st[:, j]
            print("  %20s | %10.2f | %10.2f | %10.2f | %10.2f | %10.2f" % (
                JOINT_NAMES[j], np.min(col), np.max(col), np.mean(col), np.std(col), np.ptp(col)))

    # Action-State consistency check
    if global_actions and global_states:
        print("\n  Action-State Correlation Check:")
        all_act_flat = np.concatenate(global_actions, axis=0)
        all_st_flat = np.concatenate(global_states, axis=0)
        n_min = min(len(all_act_flat), len(all_st_flat))
        for j in range(all_act_flat.shape[1]):
            corr = np.corrcoef(all_act_flat[:n_min, j], all_st_flat[:n_min, j])[0, 1]
            status = "OK" if corr > 0.8 else ("WARN" if corr > 0.5 else "BAD")
            print("    %20s: correlation=%.4f [%s]" % (JOINT_NAMES[j], corr, status))

    if constant_joints_eps:
        print("\n  LOW-VARIANCE JOINTS (std < 0.5):")
        for ep, joint, std in constant_joints_eps[:20]:
            print("    Ep %04d: %s (std=%.3f)" % (ep, joint, std))
        if len(constant_joints_eps) > 20:
            print("    ... and %d more" % (len(constant_joints_eps) - 20))

    # =====================================================================
    # [5] EPISODE DURATION & OUTLIER ANALYSIS
    # =====================================================================
    print("\n[5/6] EPISODE DURATION & OUTLIER ANALYSIS")
    if ep_frame_counts:
        fc = np.array(ep_frame_counts)
        dur = np.array(ep_durations)
        print("  Frame counts: mean=%.0f, std=%.0f, min=%d, max=%d" % (np.mean(fc), np.std(fc), np.min(fc), np.max(fc)))
        print("  Durations: mean=%.1fs, std=%.1fs, min=%.1fs, max=%.1fs" % (np.mean(dur), np.std(dur), np.min(dur), np.max(dur)))

        # Outliers > 2 std
        outliers = [(i, int(fc[i]), float(dur[i])) for i in range(len(fc)) if abs(fc[i] - np.mean(fc)) > 2 * np.std(fc)]
        if outliers:
            print("  OUTLIER EPISODES (>2 sigma from mean):")
            for ep_idx, count, d in outliers:
                direction = "TOO LONG" if count > np.mean(fc) else "TOO SHORT"
                print("    Ep %04d: %d frames (%.1fs) [%s]" % (ep_idx, count, d, direction))
        else:
            print("  No outlier episodes detected.")

    if timing_jitters:
        print("\n  Timing Jitter: mean=%.2f%%, max=%.2f%%" % (np.mean(timing_jitters), np.max(timing_jitters)))
        if np.max(timing_jitters) > 20:
            print("  WARNING: High timing jitter in some episodes!")
        else:
            print("  Timing stability: GOOD")

    # =====================================================================
    # [6] TRAIN-READINESS SUMMARY
    # =====================================================================
    print("\n[6/6] TRAIN-READINESS SUMMARY")
    print("  " + "-" * 50)

    checks = [
        ("35+ episodes recorded", n_episodes >= 35),
        ("No corrupt frames", total_corrupt == 0),
        ("No black frames", total_black == 0),
        ("Camera sync OK", sync_issues == 0),
        ("Contiguous numbering", len(missing) == 0),
        ("Action data present", len(global_actions) == n_episodes),
        ("State data present", len(global_states) == n_episodes),
    ]

    all_pass = True
    for check, passed in checks:
        icon = "[PASS]" if passed else "[FAIL]"
        print("  %s %s" % (icon, check))
        if not passed:
            all_pass = False

    print("  " + "-" * 50)
    if all_pass:
        print("  VERDICT: DATASET IS TRAIN-READY")
    else:
        print("  VERDICT: ISSUES DETECTED - REVIEW ABOVE")

    if all_issues:
        print("\n  ALL ISSUES (%d):" % len(all_issues))
        for ep, issue in all_issues:
            print("    Ep %04d: %s" % (ep, issue))

    print("\n" + "=" * 70)
    print("  AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
