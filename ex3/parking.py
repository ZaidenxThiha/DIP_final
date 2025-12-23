import argparse
from pathlib import Path

import cv2
from ultralytics import solutions


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    parser = argparse.ArgumentParser(description="Parking space occupancy demo (Ultralytics ParkingManagement)")
    parser.add_argument("-i", "--input", default=str((base_dir / "parking_video.mp4").resolve()), help="Input video path")
    parser.add_argument("--json", default=str((base_dir / "bounding_boxes.json").resolve()), help="Parking spots JSON path")
    parser.add_argument("--model", default=str((repo_root / "models" / "yolo11n.pt").resolve()), help="YOLO model weights path")
    parser.add_argument("-o", "--output", default=None, help="Optional output video path (annotated)")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f'Input video not found: "{input_path}"')
    if not json_path.exists():
        raise FileNotFoundError(f'Parking JSON not found: "{json_path}"')
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: "{model_path}"')

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f'Error reading video file: "{input_path}"')

    # Initialize
    parking_manager = solutions.ParkingManagement(
        model=str(model_path),
        json_file=str(json_path),
        tracker="bytetrack.yaml",
        classes=[2],
        iou=0.7,
        conf=0.15,
        show=False,
    )

    writer = None
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise IOError(f'Cannot open output video for writing: "{out_path}"')

    if not args.no_preview:
        cv2.namedWindow("Parking Manager", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = parking_manager(frame)
        annotated = results.plot_im
        if writer is not None:
            writer.write(annotated)
        if not args.no_preview:
            cv2.imshow("Parking Manager", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
