import cv2
import math
import time
import types
import argparse
from pathlib import Path
from collections import defaultdict, deque

from ultralytics import YOLO, solutions

# ===============================
# CONFIG
# ===============================
DEFAULT_INPUT_VIDEO = "ex1.mp4"
DEFAULT_OUTPUT_VIDEO = "final_output_heatmap.mp4"
DEFAULT_MODEL_PATH = "models/yolo11n.pt"

CONF = 0.3
IOU = 0.5
TRACKER = "botsort.yaml"
VANISH_FRAMES = 20  # how long to keep an ID after it disappears

PIXELS_PER_METER = 50.0      # px per meter (for approximate m/s)
MOTION_WINDOW = 8            # frames required before speed is locked
STOP_SPEED_M_S = 0.20        # speed threshold to show "STOP"
STOP_MIN_HISTORY = 4         # frames of history before STOP can trigger

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS = 1

# ===============================
# UTIL
# ===============================
def draw_text(img, text, pos, color=(255, 255, 255)):
    cv2.putText(img, text, pos, FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

def avg_step_distance(points) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    ) / (len(points) - 1)


class _SharedTracks:
    results = None


def _bind_shared_tracks(solution, shared: _SharedTracks) -> None:
    """
    Patch a Solution instance to reuse `shared.results` (a single YOLO track() result)
    instead of running its own internal `model.track()` call.
    """

    def _extract_tracks_from_shared(self, im0):
        r = shared.results
        self.tracks = r
        is_obb = getattr(r, "obb", None) is not None
        track_data = r.obb if is_obb else r.boxes
        self.track_data = track_data

        if track_data is not None and getattr(track_data, "is_track", False) and getattr(track_data, "id", None) is not None:
            self.boxes = (track_data.xyxyxyxy if is_obb else track_data.xyxy).cpu()
            self.clss = track_data.cls.cpu().tolist()
            self.track_ids = track_data.id.int().cpu().tolist()
            self.confs = track_data.conf.cpu().tolist()
        else:
            self.boxes, self.clss, self.track_ids, self.confs = [], [], [], []

    solution.extract_tracks = types.MethodType(_extract_tracks_from_shared, solution)


def _resolve_existing_path(path: str | None, candidates: list[Path], *, base_dirs: tuple[Path, ...] = ()) -> str:
    if path:
        p = Path(path).expanduser()
        if p.is_absolute():
            if p.exists():
                return str(p)
            raise FileNotFoundError(f"File not found: {p}")

        for base in (Path.cwd(), *base_dirs):
            candidate = (base / p).resolve()
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f'File not found: "{path}" (searched in {Path.cwd()} and {", ".join(str(b) for b in base_dirs)})')

    for c in candidates:
        if c.exists():
            return str(c)

    search_dirs = (Path.cwd(), *base_dirs)
    video_files = sorted(
        {
            p
            for d in search_dirs
            for p in (
                *d.glob("*.mp4"),
                *d.glob("*.avi"),
                *d.glob("*.mov"),
                *d.glob("*.mkv"),
            )
        }
    )
    hint = f" Available videos: {', '.join(p.name for p in video_files)}" if video_files else ""
    raise FileNotFoundError(f"No input video found.{hint}")


# ===============================
# MAIN
# ===============================
def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO tracking + heatmap + left/right occupancy + speed overlay")
    parser.add_argument("-i", "--input", default=None, help="Input video file (default: auto-detect)")
    parser.add_argument("-o", "--output", default=None, help="Output video file (default: next to this script)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Ultralytics YOLO model path")
    parser.add_argument("--stop-speed", type=float, default=STOP_SPEED_M_S, help='m/s threshold to show "STOP"')
    args = parser.parse_args()

    repo_root = base_dir.parent
    input_video = _resolve_existing_path(
        args.input,
        candidates=[
            base_dir / DEFAULT_INPUT_VIDEO,
            base_dir / "input.mp4",
            Path.cwd() / DEFAULT_INPUT_VIDEO,
            Path.cwd() / "input.mp4",
        ],
        base_dirs=(base_dir, repo_root),
    )
    model_path = _resolve_existing_path(
        args.model,
        candidates=[
            base_dir / DEFAULT_MODEL_PATH,
            repo_root / DEFAULT_MODEL_PATH,
            repo_root / "models" / Path(DEFAULT_MODEL_PATH).name,
            base_dir / Path(DEFAULT_MODEL_PATH).name,
            Path.cwd() / DEFAULT_MODEL_PATH,
            Path.cwd() / "models" / Path(DEFAULT_MODEL_PATH).name,
        ],
        base_dirs=(base_dir, repo_root),
    )
    output_video_path = Path(args.output).expanduser() if args.output else (base_dir / DEFAULT_OUTPUT_VIDEO)
    output_video_path = output_video_path.resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_video = str(output_video_path)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError(f'Cannot open input video: "{input_video}"')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    mid_x = w // 2

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    if not writer.isOpened():
        raise IOError(f'Cannot open output video for writing: "{output_video}"')

    # Ultralytics Heatmap with your desired color (TURBO)
    heatmap = solutions.Heatmap(
        model=model_path,
        classes=[0],
        conf=CONF,
        iou=IOU,
        tracker=TRACKER,
        colormap=cv2.COLORMAP_TURBO,  
        show=False,
        show_conf=False,
        show_labels=False
    )

    shared = _SharedTracks()
    _bind_shared_tracks(heatmap, shared)

    # Track-ID based region occupancy (YOLO tracking IDs)
    last_seen: dict[int, int] = {}
    last_region: dict[int, str] = {}
    track_history = defaultdict(lambda: deque(maxlen=int(MOTION_WINDOW)))
    left_count = 0
    right_count = 0
    frame_index = 0

    prev_time = time.time()

    print("Running… Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --------------------------
        # 0) TRACK ONCE (shared for all overlays)
        # --------------------------
        results = model.track(
            frame,
            persist=True,
            conf=CONF,
            iou=IOU,
            classes=[0],
            tracker=TRACKER,
            verbose=False,
        )[0]
        shared.results = results
        frame_index += 1

        # --------------------------
        # 1) HEATMAP (Ultralytics solution)
        # --------------------------
        out = heatmap.process(frame).plot_im

        # --------------------------
        # 2) LEFT/RIGHT REGION COUNT (custom; not in/out crossings)
        # --------------------------
        active_ids: set[int] = set()

        boxes = getattr(results, "boxes", None)
        use_track_ids = boxes is not None and getattr(boxes, "id", None) is not None and getattr(boxes, "xyxy", None) is not None
        if use_track_ids:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu()

            for tid, box in zip(ids, xyxy):
                cx = float((box[0] + box[2]) / 2.0)
                region = "LEFT" if cx < float(mid_x) else "RIGHT"

                active_ids.add(tid)
                last_seen[tid] = frame_index

                if tid not in last_region:
                    last_region[tid] = region
                    if region == "LEFT":
                        left_count += 1
                    else:
                        right_count += 1
                elif last_region[tid] != region:
                    if region == "LEFT":
                        left_count += 1
                        right_count -= 1
                    else:
                        right_count += 1
                        left_count -= 1
                    last_region[tid] = region

                x1, y1, x2, y2 = map(int, box.tolist())
                cy = float((box[1] + box[3]) / 2.0)
                track_history[tid].append((cx, cy))

                avg_px = avg_step_distance(list(track_history[tid]))
                speed_m_s = (avg_px * float(fps)) / float(PIXELS_PER_METER)

                cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)
                if len(track_history[tid]) >= 2:
                    if len(track_history[tid]) >= int(STOP_MIN_HISTORY) and speed_m_s <= float(args.stop_speed):
                        draw_text(out, "STOP", (x1, max(15, y1 - 6)), color=(0, 0, 255))
                    else:
                        draw_text(out, f"{speed_m_s:.2f} m/s", (x1, max(15, y1 - 6)))
        elif boxes is not None and getattr(boxes, "xyxy", None) is not None:
            # Fallback if tracking IDs are unavailable: per-frame region counts.
            xyxy = boxes.xyxy.cpu()
            if len(xyxy) > 0:
                center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                left_count = int((center_x < float(mid_x)).sum().item())
                right_count = int((center_x >= float(mid_x)).sum().item())

        if use_track_ids:
            # Cleanup vanished track IDs
            for tid in list(last_seen.keys()):
                if tid in active_ids:
                    continue
                if frame_index - last_seen[tid] <= VANISH_FRAMES:
                    continue

                if last_region.get(tid) == "LEFT":
                    left_count -= 1
                elif last_region.get(tid) == "RIGHT":
                    right_count -= 1
                last_seen.pop(tid, None)
                last_region.pop(tid, None)
                track_history.pop(tid, None)

        left_count = max(0, left_count)
        right_count = max(0, right_count)
        total_count = left_count + right_count

        # Divider line (left vs right)
        cv2.line(out, (mid_x, 0), (mid_x, h), (0, 255, 255), 2)

        # --------------------------
        # 4) HUD (FPS + COUNTS)
        # --------------------------
        now = time.time()
        fps_disp = 1.0 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        draw_text(out, f"FPS: {fps_disp:.1f}", (10, 20))
        draw_text(out, f"LEFT: {left_count}", (10, 40))
        draw_text(out, f"RIGHT: {right_count}", (mid_x + 10, 40))
        draw_text(out, f"TOTAL: {total_count}", (10, 60))

        writer.write(out)
        cv2.imshow("Heatmap + L/R Count + Speed (m/s)", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
