"""Standalone script to print FPS and frame dimensions for an MP4 video."""

import argparse
import os

try:
    import cv2
except Exception as exc:
    raise SystemExit(f"[ERROR] OpenCV is required (pip install opencv-python). Details: {exc}")


def main(default_video_path=None) -> None:
    parser = argparse.ArgumentParser(
        description="Print video frame rate and frame dimensions for an MP4 file."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=default_video_path,
        help="Path to the input .mp4 video",
    )
    args = parser.parse_args()

    video_path = args.video_path
    if not video_path:
        raise SystemExit("[ERROR] Missing video path. Pass it as an argument or set default_video_path.")
    if not os.path.exists(video_path):
        raise SystemExit(f"[ERROR] Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Dimensions: {width}x{height} (width x height)")
    print(f"Frame count: {frame_count}")


if __name__ == "__main__":
    # Option 1: leave None and pass path in command line
    # Option 2: set a default path string here
    main(default_video_path=r"C:\Users\Utilizador\Carey Lab Dropbox\Alice Geminiani\LocoCF-Data\Data\Optogenetics\Experiments JAWS RT\Tied belt sessions\ALL ANIMALS\VIV42430_111_27_0.275_0.275_tied_1_1.mp4")
