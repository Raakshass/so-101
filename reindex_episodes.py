"""
Re-index episodes after deleting episode 0005.
Renames episode_0006 -> episode_0005, episode_0007 -> episode_0006, etc.
Also updates episode_num inside each episode_data.json.
"""
import os
import sys
import json
import shutil
from pathlib import Path

DATASET_DIR = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "lerobot" / "siddhantjain" / "pick_bottle_training"

def main():
    # Step 1: Delete episode 0005
    ep5_dir = DATASET_DIR / "episode_0005"
    if ep5_dir.exists():
        shutil.rmtree(ep5_dir)
        print("DELETED: episode_0005 (128.4s outlier)")
    else:
        print("episode_0005 already deleted")

    # Step 2: Get remaining episodes sorted
    episode_dirs = sorted(DATASET_DIR.glob("episode_*"))
    print("Remaining episodes: %d" % len(episode_dirs))

    # Step 3: Re-index
    renames = []
    for new_idx, ep_dir in enumerate(episode_dirs):
        old_num = int(ep_dir.name.split("_")[1])
        if old_num != new_idx:
            new_name = "episode_%04d" % new_idx
            renames.append((ep_dir, DATASET_DIR / new_name, old_num, new_idx))

    if not renames:
        print("No re-indexing needed.")
    else:
        # Rename to temp names first to avoid collisions
        temp_renames = []
        for old_path, new_path, old_num, new_idx in renames:
            temp_path = DATASET_DIR / ("_temp_episode_%04d" % new_idx)
            os.rename(old_path, temp_path)
            temp_renames.append((temp_path, new_path, new_idx))
            print("  %s -> %s (temp)" % (old_path.name, temp_path.name))

        # Now rename from temp to final
        for temp_path, new_path, new_idx in temp_renames:
            os.rename(temp_path, new_path)
            # Update episode_data.json
            json_path = new_path / "episode_data.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    data = json.load(f)
                data["episode_num"] = new_idx
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2)
            print("  %s -> %s (final, json updated)" % (temp_path.name, new_path.name))

    # Step 4: Verify
    final_dirs = sorted(DATASET_DIR.glob("episode_*"))
    print("\nFinal state: %d episodes" % len(final_dirs))
    ep_nums = [int(d.name.split("_")[1]) for d in final_dirs]
    expected = list(range(len(final_dirs)))
    if ep_nums == expected:
        print("Numbering: CONTIGUOUS [0-%d]" % (len(final_dirs) - 1))
    else:
        print("ERROR: numbering gap! Got: %s" % ep_nums)

    # Verify frame counts
    total_frames = 0
    for d in final_dirs:
        json_path = d / "episode_data.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            total_frames += data.get("num_frames", 0)
    print("Total frames: %d" % total_frames)
    print("\nDone. Ready to record more episodes starting from episode_%04d" % len(final_dirs))

if __name__ == "__main__":
    main()
