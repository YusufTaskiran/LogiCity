import os
import shutil
import pickle as pkl
import argparse
from pathlib import Path

from tools.pkl2city import main as render_episode
from tools.img2video import create_gif


def infer_ego_id(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    agents = data.get("Static Info", {}).get("Agents", {})
    if not agents:
        return 3
    car_agents = [
        info["layer_id"]
        for info in agents.values()
        if info.get("type") == "Car"
    ]
    if car_agents:
        return min(car_agents)
    return min(info["layer_id"] for info in agents.values())


def parse_ego_ids(ego_ids):
    if ego_ids is None:
        return None
    if isinstance(ego_ids, (list, tuple)):
        return [int(v) for v in ego_ids]
    return [int(v) for v in str(ego_ids).split(",") if str(v).strip()]


def is_visualizable_world_pkl(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
        return isinstance(data, dict) and "Time_Obs" in data and "Static Info" in data
    except Exception:
        return False


def batch_render(input_dir, output_dir, ego_id=None, ego_ids=None, keep_frames=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(input_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}")

    for pkl_file in pkl_files:
        if not is_visualizable_world_pkl(pkl_file):
            print(f"Skipping non-visualizable pickle: {pkl_file}")
            continue
        base_name = pkl_file.stem
        temp_frames = output_dir / f"{base_name}_frames"
        gif_path = output_dir / f"{base_name}.gif"
        local_ego_ids = parse_ego_ids(ego_ids)
        if local_ego_ids is None:
            if ego_id is not None:
                local_ego_ids = [int(ego_id)]
            else:
                local_ego_ids = [infer_ego_id(pkl_file)]

        print(f"Rendering {pkl_file} -> {gif_path} (ego_ids={local_ego_ids})")
        temp_frames.mkdir(parents=True, exist_ok=True)
        render_episode(str(pkl_file), local_ego_ids, str(temp_frames))
        create_gif(str(temp_frames), str(gif_path))

        if not keep_frames:
            shutil.rmtree(temp_frames, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render episode world pickles to GIFs.")
    parser.add_argument("--input_dir", required=True, help="Directory containing visualizable episode .pkl files.")
    parser.add_argument("--output_dir", required=True, help="Directory where GIFs will be written.")
    parser.add_argument("--ego_id", type=int, default=None, help="Optional fixed ego layer id. If omitted, infer from pickle.")
    parser.add_argument("--ego_ids", type=str, default=None, help="Optional comma-separated ego layer ids, e.g. '3,4'.")
    parser.add_argument("--keep_frames", action="store_true", help="Keep intermediate frame folders.")
    args = parser.parse_args()

    batch_render(args.input_dir, args.output_dir, ego_id=args.ego_id, ego_ids=args.ego_ids, keep_frames=args.keep_frames)
