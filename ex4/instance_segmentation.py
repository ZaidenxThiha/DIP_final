import argparse
from pathlib import Path

import cv2
from ultralytics import solutions


def main():
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    parser = argparse.ArgumentParser(description="Instance segmentation demo (Ultralytics InstanceSegmentation)")
    parser.add_argument("-i", "--input", default=str((base_dir / "car_video.mp4").resolve()), help="Input video path")
    parser.add_argument("--model", default=str((repo_root / "models" / "yolo11n-seg.pt").resolve()), help="YOLO segmentation weights path")
    parser.add_argument("-o", "--output", default=None, help="Optional output video path (annotated)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f'Input video not found: "{input_path}"')
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: "{model_path}"')

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f'Error reading video source: "{input_path}"')

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
        cv2.namedWindow("Instance Segmentation", cv2.WINDOW_NORMAL)

    isegment = solutions.InstanceSegmentation(
        model=str(model_path),
        conf=float(args.conf),
        iou=float(args.iou),
        tracker="bytetrack.yaml",
        classes=[2],
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = isegment(frame)
        annotated_frame = results.plot_im
        if writer is not None:
            writer.write(annotated_frame)
        if not args.no_preview:
            cv2.imshow("Instance Segmentation", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
