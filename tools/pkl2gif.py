import argparse
import os
from glob import glob
from PIL import Image

from pkl2city import main as render_pkl_to_frames


def _frame_index(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem.split("_")[-1])


def build_gif_from_folder(image_folder, output_gif, duration_ms=150):
    image_files = sorted(glob(os.path.join(image_folder, "step_*.png")), key=_frame_index)
    if not image_files:
        raise RuntimeError("No rendered frames found in {}.".format(image_folder))

    frames = [Image.open(path).convert("RGB") for path in image_files]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Render a cached world PKL to frames and assemble a GIF.")
    parser.add_argument("--pkl", required=True, help="Path to the cached world PKL file.")
    parser.add_argument("--ego_id", type=int, default=3, help="Ego agent layer id for start/goal markers.")
    parser.add_argument("--output_folder", required=True, help="Folder for rendered PNG frames.")
    parser.add_argument("--output_gif", default=None, help="Output GIF path. Defaults to <output_folder>.gif")
    parser.add_argument("--duration_ms", type=int, default=150, help="GIF frame duration in milliseconds.")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Upscale rendered frames by this factor before GIF assembly.")
    parser.add_argument("--crop_size", type=int, default=None, help="Optional top-left square crop size. Defaults to full frame.")
    parser.add_argument("--max_step", type=int, default=None, help="Optional maximum timestep to render into the GIF.")
    args = parser.parse_args()

    render_pkl_to_frames(
        args.pkl,
        args.ego_id,
        args.output_folder,
        scale_factor=args.scale_factor,
        crop_size=args.crop_size,
        max_step=args.max_step,
    )

    output_gif = args.output_gif
    if output_gif is None:
        normalized = os.path.normpath(args.output_folder)
        output_gif = "{}.gif".format(normalized)

    build_gif_from_folder(args.output_folder, output_gif, duration_ms=args.duration_ms)
    print("Saved GIF to {}".format(output_gif))


if __name__ == "__main__":
    main()
