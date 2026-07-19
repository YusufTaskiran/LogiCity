import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Frame size mismatch for {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    main()
