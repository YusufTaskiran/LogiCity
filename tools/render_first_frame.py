import argparse
import os
import pickle as pkl

import cv2

from tools.batch_visualize_episodes import infer_ego_ids
from tools.pkl2city import ICON_SIZE_DICT, PATH_DICT, gridmap2img_agents, gridmap2img_static, resize_with_aspect_ratio


def load_icons():
    icon_dict = {}
    for key, path_or_paths in PATH_DICT.items():
        if isinstance(path_or_paths, list):
            raw_imgs = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in path_or_paths]
            icon_dict[key] = [resize_with_aspect_ratio(img, ICON_SIZE_DICT[key]) for img in raw_imgs]
        else:
            raw_img = cv2.cvtColor(cv2.imread(path_or_paths), cv2.COLOR_BGR2RGB)
            icon_dict[key] = resize_with_aspect_ratio(raw_img, ICON_SIZE_DICT[key])
    return icon_dict


def render_frame(pkl_path, output_path, ego_ids=None, frame_index=0):
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    obs = data["Time_Obs"]
    agents = data["Static Info"]["Agents"]
    time_steps = sorted(obs.keys())
    if not time_steps:
        raise ValueError(f"No Time_Obs found in {pkl_path}")

    if ego_ids is None:
        ego_ids = infer_ego_ids(pkl_path)

    icon_dict = load_icons()
    frame_index = max(0, min(int(frame_index), len(time_steps) - 1))
    current_key = time_steps[frame_index]
    next_key = time_steps[frame_index + 1] if frame_index + 1 < len(time_steps) else current_key

    grid = obs[current_key]["World"].numpy()
    next_grid = obs[next_key]["World"].numpy()
    static_map = gridmap2img_static(grid, icon_dict, ego_ids)
    img, _ = gridmap2img_agents(grid, next_grid, icon_dict, static_map, last_icons=None, agents=agents, ego_ids=ego_ids)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render a specific frame of a rollout pickle, without step text.")
    parser.add_argument("--pkl", required=True, help="Path to a visualizable rollout/world pickle.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--ego_ids", default=None, help="Optional comma-separated ego layer ids, e.g. '3,4'.")
    parser.add_argument("--frame_index", type=int, default=0, help="Zero-based frame index to render. Use 2 for the 3rd frame.")
    args = parser.parse_args()

    ego_ids = None
    if args.ego_ids:
        ego_ids = [int(x) for x in args.ego_ids.split(",") if x.strip()]

    render_frame(args.pkl, args.output, ego_ids=ego_ids, frame_index=args.frame_index)
    print(args.output)


if __name__ == "__main__":
    main()
