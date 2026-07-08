# RIG HANDOFF — SO-100 Shape-Sorting Phase (laptop → RTX 5090 rig)

**Date:** 2026-07-08 · **From:** Windows laptop (RTX 3050 4GB) · **To:** PC rig (RTX 5090 32GB)
**Owner:** Raakshass (HF) / Siddhant Jain · **Repo:** https://github.com/Raakshass/so-101
**Codebase:** LeRobot **v0.6.0** (upstream release tag merged 2026-07-08; includes the MolmoAct2 SO-100/101 calibration correction, PR #3879)

This file is the source of truth for continuing the project on the rig.
**Agent instruction: verify every command against `src/lerobot` before running — this repo IS the LeRobot install (editable), so the code on disk is definitive.**

---

## 1. What is already done (Phase 1 — complete ✅)

- 56 teleop episodes recorded (bottle pick-and-place), SmolVLA fine-tuned → **`Raakshass/smolvla_so100_pick_bottle`** on HF Hub.
- Deployed successfully on the physical SO-100 with:
  `python -m lerobot.scripts.lerobot_rollout --strategy.type=base --inference.type=rtc ...`
- The exact working launcher is **`run_rollout.ps1`** in this repo — read it; every flag in it is a lesson learned.

## 2. Hardware identity (travels with the arms — machine-independent)

| Item | Stable identifier | On old laptop | On this rig |
|---|---|---|---|
| Follower arm (SO-100, Feetech STS3215) | USB serial **5A7C120825** | COM12 | **re-detect** |
| Leader arm | USB serial **5AAF288338** | COM11 | **re-detect** |
| Overhead camera → `cam_high` | visual: sees table/workspace top-down | index 1 (DSHOW) | **re-probe** |
| Wrist camera → `cam_wrist` | visual: mounted on gripper | index 2 (DSHOW) | **re-probe** |
| Robot calibration id | `my_follower` | — | restore (below) |
| Teleop calibration id | `my_leader` | — | restore (below) |

- Find ports: `python -m lerobot.scripts.lerobot_find_port` (or `scan_motors.py` / `scan_leader.py`).
- Find cameras: `python -m lerobot.scripts.lerobot_find_cameras` + `camera_verify_labeled.py` for labeled snapshots. **Confirm the mapping visually (overhead vs wrist) — never trust index numbers.**

## 3. Environment setup on the rig

```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
# RTX 5090 is Blackwell (sm_120): install CUDA 12.8 wheels or newer, or torch will
# fail with "no kernel image is available for execution on the device".
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/Raakshass/so-101.git && cd so-101
pip install -e ".[feetech]"
pip install num2words            # required by SmolVLM/SmolVLA processor
huggingface-cli login            # account: Raakshass (needed to push datasets/models)
```

**Version policy:** this fork is pinned to the **LeRobot v0.6.0 release** (verify: `python -c "import lerobot; print(lerobot.__version__)"` from OUTSIDE the repo dir → `0.6.0`). To pull future upstream releases:
```bash
git remote add upstream https://github.com/huggingface/lerobot.git   # if not present
git fetch upstream tag vX.Y.Z && git merge vX.Y.Z && pip install -e ".[feetech]"
```
Re-read the release notes and `docs/source/backwardcomp.mdx` before syncing — calibration conventions have changed across LeRobot versions before.

## 4. Restore calibration (BEFORE first connect)

PowerShell (Windows rig):
```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so_follower" | Out-Null
Copy-Item calibration_backup\robots\so_follower\my_follower.json "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\robots\so_follower\"
New-Item -ItemType Directory -Force "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\teleoperators\so_leader" | Out-Null
Copy-Item calibration_backup\teleoperators\so_leader\my_leader.json "$env:USERPROFILE\.cache\huggingface\lerobot\calibration\teleoperators\so_leader\"
```
bash (Linux rig):
```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
cp calibration_backup/robots/so_follower/my_follower.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp calibration_backup/teleoperators/so_leader/my_leader.json ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```
Always pass `--robot.id=my_follower` / `--teleop.id=my_leader`. If `connect()` prompts about a calibration mismatch, press ENTER to use the file — only re-calibrate if teleop motion is visibly wrong.

## 5. Hard-won gotchas (each one cost real debugging time)

1. **`--fps` at rollout must equal the dataset fps used for training.** Phase-1 data was 15 fps; new data should be 30 fps (see §7) → rollout with `--fps=30`.
2. **RTC is mandatory for VLA models**, and `--inference.rtc.execution_horizon` must exceed the measured inference delay. Watch the log line `on_queue.py: ... real_delay=N` — steady-state N is your latency in control ticks. Set horizon ≈ 2×N (must stay < the policy `chunk_size`: 30 for MolmoAct2, 50 for SmolVLA). Symptom of horizon too small: the arm "vibrates" — executes a few actions, stalls, jumps. (On the 3050, SmolVLA measured real_delay 18–21 at 15 fps; horizon 10 starved → vibration; horizon 30 → smooth.)
3. **Windows cameras: always `backend: DSHOW`** in `--robot.cameras` (default MSMF is flaky). Linux: omit backend.
4. **Windows console: set `PYTHONIOENCODING=utf-8`** (and `PYTHONUTF8=1`) or logging with `≥` crashes with a cp1252 UnicodeEncodeError.
5. **If HF Hub flakes mid-load** (`Cannot send a request, as the client has been closed`), set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` after the first successful download — everything loads from cache.
6. **Never run `python -c "import lerobot..."` from the repo root** — `src/lerobot/types.py` shadows the stdlib `types` module in some path setups. Run scripts as files or `-m` modules; run `-c` snippets from another directory.
7. `--display_data=true` needs the Rerun viewer on PATH (`rerun.exe` lives in the env's `Scripts/`; activate the env or prepend it to PATH).
8. The board/base must be **fixed to the table** (tape an outline) for the entire data-collection + eval campaign. Moving it = new data distribution.

## 6. New task — shape-sorter (Phase 2)

Wooden board with 4 peg clusters (1, 2, 3, 4 pegs). Four shapes — **circle (1 hole, rotation-invariant), rectangle (2 holes, 180° symmetry), triangle (3 holes, 120°), square (4 holes, 90°)** — must be picked, oriented, and threaded onto their matching peg cluster. Hole-to-peg tolerance is ~1–2 mm, at the edge of SO-100 repeatability → **hardware feasibility is gate #1, before any training.**

### Curriculum (data-collection stages; training is always on the cumulative mixture)

| Stage | Data to collect | Prompt pattern | Gate to advance |
|---|---|---|---|
| **0. Feasibility** | none — human teleop only, 10–20 insertion attempts per shape | — | ≥80% human success per shape; else fix physics (chamfer/sand pegs, adjust wrist-cam view of peg tips, board fixation) |
| **1. Circle** | 50–60 eps, varied positions (skip fixed-position stage — waste of teleop) | "pick up the circle and put it on the circle pegs" | ACT sanity model ≥70% on 10 trials |
| **2. Per-shape** | +80–100 eps each: rectangle, square, triangle, random position AND rotation | "pick up the {shape} and put it on the {shape} pegs" | ≥60% per shape (VLA on cumulative data) |
| **3. Selection** | +100–150 eps: 2–3 shapes on table, prompt names the target | same prompts, distractors present | ≥60% correct-pick+insert |
| **4. Full sort** | +150–200 eps: all 4 shapes, sequential sort | "sort all the shapes onto their pegs" | ≥50% full completion |

**Total budget ≈ 550–650 episodes, collected adaptively** — expand a stage only where the eval shows failures. Do not pre-commit to more.
**Demo quality rule:** the policy ceiling = demonstrator ceiling. Demos should slow down near insertion, use a consistent strategy (hover above pegs → align → vertical descend → slight wiggle on contact), and never include failed/corrected grabs unless you re-record.

## 7. Recording (native LeRobot recorder — do NOT use the old custom JPEG pipeline)

```powershell
python -m lerobot.scripts.lerobot_record `
    --robot.type=so100_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower `
    --robot.cameras="{ cam_high: {type: opencv, index_or_path: <HIGH_IDX>, width: 640, height: 480, fps: 30, backend: DSHOW}, cam_wrist: {type: opencv, index_or_path: <WRIST_IDX>, width: 640, height: 480, fps: 30, backend: DSHOW}}" `
    --teleop.type=so100_leader --teleop.port=<LEADER_PORT> --teleop.id=my_leader `
    --dataset.repo_id=Raakshass/so100_shape_sorting `
    --dataset.single_task="pick up the circle and put it on the circle pegs" `
    --dataset.num_episodes=60 --dataset.fps=30 `
    --dataset.episode_time_s=60 --dataset.reset_time_s=15 `
    --display_data=true
```
30 fps (not 15) for finer motion resolution during precision insertion. Keep camera names exactly `cam_high` / `cam_wrist` for continuity with Phase 1.

## 8. Training on the 5090 (all native in this repo — verified in `src/lerobot/policies/`)

**a) ACT sanity baseline (hours, run first on Stage-1 data):**
```bash
python -m lerobot.scripts.lerobot_train \
    --policy.type=act --dataset.repo_id=Raakshass/so100_shape_sorting \
    --output_dir=outputs/act_shapes --job_name=act_shapes \
    --policy.device=cuda --batch_size=8 --steps=100000 --policy.push_to_hub=false
```

**b) SmolVLA fine-tune (fast iteration, known-good from Phase 1):**
```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/smolvla_base --dataset.repo_id=Raakshass/so100_shape_sorting \
    --output_dir=outputs/smolvla_shapes --job_name=smolvla_shapes \
    --policy.device=cuda --batch_size=32 --steps=30000
```

**c) MolmoAct2 LoRA (capability ceiling — 7B, needs the 5090):**

⚠️ **Use the SO-100-corrected checkpoint as the base**: `lerobot/MolmoAct2-SO100_101-LeRobot` (verified on Hub). The raw MolmoAct2-SO100_101 weights were trained under an older joint-calibration convention than LeRobot ≥ 0.5.0 — **without the frame correction the arm moves in the wrong direction**. The `lerobot/...-LeRobot` conversion has the correction built into its processor pipeline; if you ever start from the raw checkpoint instead, you must set `--policy.joint_signs="[1,-1,1,1,1,1]" --policy.joint_offsets="[0,90,90,0,0,0]"`. Full details: `docs/source/molmoact2.mdx` → "Hardware Deployment" (added in PR #3879, included in this repo).

```bash
python -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot \
    --dataset.repo_id=Raakshass/so100_shape_sorting \
    --policy.enable_lora_vlm=true --policy.enable_lora_action_expert=true \
    --policy.gradient_checkpointing=true \
    --output_dir=outputs/molmoact2_shapes --job_name=molmoact2_shapes \
    --policy.device=cuda --batch_size=4 --steps=30000 --policy.push_to_hub=false
```
Generic base (non-SO-100 embodiments): `--policy.type=molmoact2` uses default `checkpoint_path="allenai/MolmoAct2"` (exists on Hub).
Verified config facts (`configuration_molmoact2.py`): `chunk_size=30`, `model_dtype="bfloat16"` default, `lora_rank=64` default, `rtc_config` supported.
**Constraints enforced by the config:** `enable_lora_action_expert` requires `enable_lora_vlm=true`; `train_action_expert_only=true` is **incompatible** with `enable_lora_vlm` and requires `action_mode="continuous"`. (An earlier analysis suggested combining them — the config will refuse.)
**Bonus:** the SO-100 checkpoint can be tried **zero-shot** on the shape task before any training (`lerobot_rollout --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot`) — expect low success, but it validates the whole inference path and gives a baseline for free.

## 9. Deployment / evaluation

```powershell
python -m lerobot.scripts.lerobot_rollout `
    --strategy.type=base `
    --policy.path=<outputs/.../checkpoints/last/pretrained_model or hub id> `
    --inference.type=rtc --inference.rtc.execution_horizon=16 `
    --robot.type=so100_follower --robot.port=<PORT> --robot.id=my_follower `
    --robot.cameras="<same string as recording>" `
    --task="pick up the circle and put it on the circle pegs" `
    --fps=30 --duration=60 --display_data=true
```
First run: watch `real_delay=N` in the logs, then set `execution_horizon ≈ 2×N` (< chunk_size). On a 5090, expect N ≈ 4–10 at 30 fps for MolmoAct2 → horizon 16–20; SmolVLA will be much lower.

**Eval protocol (use it every time, or progress is unmeasurable):** 10 trials per condition, randomized start pose per trial, binary success = shape fully seated on correct pegs within 60 s. Log a table: stage × shape × model → success rate.

## 10. Assets

- Phase-1 model: `Raakshass/smolvla_so100_pick_bottle` (Hub)
- Phase-1 raw episodes: laptop-only at `~/.cache/huggingface/lerobot/siddhantjain/pick_bottle_training` (not on Hub; push from the laptop if ever needed for co-training)
- Calibration backups: `calibration_backup/` in this repo (verified identical to the live cache on 2026-07-08)
- Proven rollout launcher: `run_rollout.ps1`
- Camera tooling: `camera_verify_labeled.py`, `camera_probe.py`, `camera_capture_all.py`
- Legacy (Phase-1 era, reference only): `record_episode.py`, `convert_to_lerobot.py`, audit scripts
