import argparse
from pathlib import Path

import cv2
from ultralytics import solutions


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No file found. Tried: {', '.join(str(p) for p in paths)}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    default_input = _first_existing(
        [
            base_dir / "parking_vdo.mp4",
            base_dir / "parking_video.mp4",
        ]
    )
    default_json = _first_existing([base_dir / "bounding_boxes.json"])
    default_model = _first_existing(
        [
            repo_root / "models" / "yolo11n.pt",
            base_dir / "yolo11n.pt",
        ]
    )

    parser = argparse.ArgumentParser(description="Parking occupancy demo (Ultralytics ParkingManagement)")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input video path")
    parser.add_argument("--json", default=str(default_json), help="Parking spots JSON path")
    parser.add_argument("--model", default=str(default_model), help="YOLO model weights path")
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

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video resolution: {w} x {h}")

    parking_manager = solutions.ParkingManagement(
        model=str(model_path),
        json_file=str(json_path),
        tracker="bytetrack.yaml",
        classes=[2],
        conf=0.15,
        iou=0.5,
        show=False,
    )
    parking_manager.occ = (0, 0, 255)  # occupied: red
    parking_manager.arc = (0, 255, 0)  # available: green

    writer = None
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
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

