import argparse
import os
import pickle as pkl

import cv2
from PIL import Image

from tools.pkl2city import ICON_SIZE_DICT, PATH_DICT, gridmap2img_static, resize_with_aspect_ratio


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


def render_static_map(pkl_path, output_path):
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)

    obs = data["Time_Obs"]
    time_steps = sorted(obs.keys())
    if len(time_steps) == 0:
        raise ValueError(f"No Time_Obs found in {pkl_path}")

    first_world = obs[time_steps[0]]["World"].numpy()
    static_map = gridmap2img_static(first_world, load_icons(), ego_ids=None)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(static_map).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Render a static thesis map without agents from a saved rollout pickle.")
    parser.add_argument("--pkl", required=True, help="Path to a rollout/world pickle containing Time_Obs.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    render_static_map(args.pkl, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
