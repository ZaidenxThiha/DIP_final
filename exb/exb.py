import csv
import cv2
import datetime as _dt
import io
import os
import time
import subprocess
import numpy as np
import pathlib
import types
import argparse
from pathlib import Path
from collections import defaultdict

from ultralytics import YOLO, solutions


# ===============================
# CONFIGURATION
# ===============================
_BASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BASE_DIR.parent

OUTPUT_VIDEO = str((_BASE_DIR / "traffic_analysis_heatmap.mp4").resolve())
RECORDS_DIR = str((_BASE_DIR / "records").resolve())
MODEL_PATH = str((_REPO_ROOT / "models" / "yolo11n.pt").resolve())
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=muijHPW82vI"

CONF = 0.25
IOU = 0.60
INFERENCE_SIZE = 1280
TRACKER = "botsort.yaml"

# Physics
PIXELS_PER_METER = 35.0
MOTION_WINDOW = 10
STOP_SPEED_THRESHOLD = 2.0

# Heatmap
HEATMAP_DECAY = 0.995
CONGESTION_THRESHOLD = 150

VEHICLE_CLASSES = [1, 2, 3, 5, 7]


# ===============================
# UTILITIES
# ===============================
def extract_youtube_stream(url):
    """Extract direct m3u8 stream using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--js-runtime", "node",
        "-f", "best[protocol*=m3u8]/best",
        "-g", url
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return next((line for line in out.splitlines() if line.startswith("http")), None)
    except Exception as e:
        print(f"Error extracting stream: {e}")
        return None


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


def _enable_live_speed_estimation(speed_estimator) -> None:
    """
    Patch a SpeedEstimator instance to compute speeds continuously (not "lock once"),
    and to annotate every visible tracked vehicle.
    """

    def _process_live(self, im0):
        from collections import deque

        from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
        from ultralytics.utils.plotting import colors

        self.frame_count += 1
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        if not hasattr(self, "trk_hist"):
            self.trk_hist = {}
        if not hasattr(self, "spd"):
            self.spd = {}

        active_ids = set()
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            active_ids.add(int(track_id))
            self.store_tracking_history(track_id, box)

            if track_id not in self.trk_hist:
                self.trk_hist[track_id] = deque(maxlen=self.max_hist)
            self.trk_hist[track_id].append(self.track_line[-1])

            speed_kmh = None
            if len(self.trk_hist[track_id]) >= 2:
                p0 = self.trk_hist[track_id][0]
                p1 = self.trk_hist[track_id][-1]
                dt = (len(self.trk_hist[track_id]) - 1) / float(self.fps)
                if dt > 0:
                    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                    pixel_distance = (dx * dx + dy * dy) ** 0.5
                    meters = pixel_distance * float(self.meter_per_pixel)
                    speed_kmh = min((meters / dt) * 3.6, float(self.max_speed))

            if speed_kmh is not None:
                self.spd[int(track_id)] = int(speed_kmh)

            label = f"{self.spd.get(int(track_id), 0)} km/h"
            annotator.box_label(box, label=label, color=colors(int(cls), True))

        # prune histories and speeds for tracks no longer visible
        self.trk_hist = {tid: h for tid, h in self.trk_hist.items() if int(tid) in active_ids}
        self.spd = {tid: v for tid, v in self.spd.items() if int(tid) in active_ids}

        plot_im = annotator.result()
        self.display_output(plot_im)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids))

    # Monkeypatch method
    speed_estimator.process = types.MethodType(_process_live, speed_estimator)


# ===============================
# FRAME PROCESSING
# ===============================
def process_frame(
    frame,
    *,
    model,
    yolo_heatmap,
    speed_estimator,
    object_counter,
    shared_tracks,
    fps,
    conf=CONF,
    iou=IOU,
    imgsz=INFERENCE_SIZE,
    draw_stop_moving=True,
    stats_out=None,
):
    """Run YOLO tracking once, then render Ultralytics Solutions (Heatmap + SpeedEstimator + ObjectCounter)."""
    stopped_ids = set()
    moving_ids = set()

    results = model.track(
        frame,
        persist=True,
        conf=conf,
        iou=iou,
        classes=VEHICLE_CLASSES,
        imgsz=imgsz,
        tracker=TRACKER,
        verbose=False,
    )[0]

    shared_tracks.results = results

    # Render pipeline (all solutions reuse the shared tracking results; no extra inference)
    frame_out = yolo_heatmap.process(frame).plot_im
    frame_out = speed_estimator.process(frame_out).plot_im
    frame_out = object_counter.process(frame_out).plot_im

    # Stop/moving classification: read speed computed by Ultralytics SpeedEstimator (km/h)
    speed_threshold_kmh = float(STOP_SPEED_THRESHOLD) * 3.6
    heat_arr = getattr(yolo_heatmap, "heatmap", None)
    visible_count = 0

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.int().cpu().tolist()
        visible_count = len(ids)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            speed_kmh = speed_estimator.spd.get(int(track_id))
            if speed_kmh is None:
                moving_ids.add(int(track_id))
                continue

            heat_value = 0.0
            if heat_arr is not None:
                hh, ww = heat_arr.shape[:2]
                if 0 <= cy < hh and 0 <= cx < ww:
                    heat_value = float(np.mean(heat_arr[cy, cx]))

            is_stopped = float(speed_kmh) < speed_threshold_kmh
            is_congested = heat_value > CONGESTION_THRESHOLD

            if is_stopped:
                stopped_ids.add(int(track_id))
                cv2.putText(
                    frame_out,
                    f"STOP ID:{int(track_id)}",
                    (int(x1), max(0, int(y1) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                if is_congested:
                    cv2.putText(
                        frame_out,
                        f"JAM {int(heat_value)}",
                        (int(x1), max(0, int(y1) - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
            else:
                moving_ids.add(int(track_id))

    if stats_out is not None:
        stats_out["stopped_ids"] = stopped_ids
        stats_out["moving_ids"] = moving_ids
        stats_out["visible_count"] = int(visible_count)

    cv2.putText(
        frame_out,
        f"Mode: Ultralytics Solutions (Heatmap+Speed+Count) | {imgsz}px",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    if draw_stop_moving:
        total_processed = int(object_counter.in_count) + int(object_counter.out_count)
        cv2.putText(
            frame_out,
            f"Visible: {visible_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_out,
            f"Stopped: {len(stopped_ids)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_out,
            f"Moving: {len(moving_ids)}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_out,
            f"Total processed: {total_processed}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame_out, len(stopped_ids), len(moving_ids)


# ===============================
# CLI (OpenCV I/O + YOLO11n inference)
# Note: OpenCV here is only used for video capture/encoding and drawing/preview.
# Object detection + tracking comes from Ultralytics YOLO (`MODEL_PATH`).
# ===============================
def run_cli():
    parser = argparse.ArgumentParser(description="Traffic heatmap from a YouTube stream (Ultralytics Solutions)")
    parser.add_argument("--url", default=None, help="YouTube URL (if omitted, prompts)")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model weights path")
    parser.add_argument("-o", "--output", default=OUTPUT_VIDEO, help="Output video path")
    args = parser.parse_args()

    model_path = str(Path(args.model).expanduser().resolve())
    print(f"Loading model {model_path}...")
    model = YOLO(model_path)

    url = (args.url or input("Enter YouTube URL: ")).strip()
    source = extract_youtube_stream(url)
    if not source:
        print("❌ Failed to extract stream.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Failed to open stream.")
        return

    ret, frame = cap.read()
    if not ret:
        print("❌ No frames received.")
        return

    H, W = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    shared = _SharedTracks()

    # Ultralytics native solutions (all reusing the same YOLO tracking result each frame)
    yolo_heatmap = solutions.Heatmap(
        model=model_path,
        classes=VEHICLE_CLASSES,
        colormap=cv2.COLORMAP_TURBO,
        conf=CONF,
        iou=IOU,
        tracker=TRACKER,
        show=False,
        show_conf=False,
        show_labels=False,
    )

    speed_estimator = solutions.SpeedEstimator(
        model=model_path,
        classes=VEHICLE_CLASSES,
        conf=CONF,
        iou=IOU,
        tracker=TRACKER,
        show=False,
        fps=float(fps),
        max_hist=int(MOTION_WINDOW),
        meter_per_pixel=float(1.0 / PIXELS_PER_METER),
        max_speed=180,
    )
    _enable_live_speed_estimation(speed_estimator)

    count_line_y = int(H * 0.60)
    object_counter = solutions.ObjectCounter(
        model=model_path,
        classes=VEHICLE_CLASSES,
        conf=CONF,
        iou=IOU,
        tracker=TRACKER,
        show=False,
        show_conf=False,
        show_labels=False,
        show_in=False,
        show_out=False,
        region=[(0, count_line_y), (W, count_line_y)],  # horizontal counting line
    )

    _bind_shared_tracks(yolo_heatmap, shared)
    _bind_shared_tracks(speed_estimator, shared)
    _bind_shared_tracks(object_counter, shared)

    print(f" Processing at {INFERENCE_SIZE}px inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream interrupted. Reconnecting...")
            time.sleep(2)
            cap = cv2.VideoCapture(source)
            continue

        frame_with_heatmap, _, _ = process_frame(
            frame,
            model=model,
            yolo_heatmap=yolo_heatmap,
            speed_estimator=speed_estimator,
            object_counter=object_counter,
            shared_tracks=shared,
            fps=fps,
            conf=CONF,
            iou=IOU,
            imgsz=INFERENCE_SIZE,
        )

        cv2.imshow("Smart Traffic Heatmap", frame_with_heatmap)
        writer.write(frame_with_heatmap)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ===============================
# Streamlit (Ultralytics live inference-style UI)
# ===============================
def _running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="Traffic Heatmap (YOLO + Ultralytics Heatmap)", layout="wide")
    st.title("Vehicle Activity Detection")

    weight_paths = sorted(
        {
            *Path.cwd().glob("*.pt"),
            *_BASE_DIR.glob("*.pt"),
            *(_REPO_ROOT / "models").glob("*.pt"),
        }
    )
    if Path(MODEL_PATH).exists():
        weight_paths = sorted({*weight_paths, Path(MODEL_PATH)})
    weights = [str(p) for p in weight_paths]
    default_weight = MODEL_PATH if MODEL_PATH in weights else (weights[0] if weights else MODEL_PATH)

    with st.sidebar:
        st.header("Settings")
        weight_path = st.selectbox("Model", options=weights or [MODEL_PATH], index=(weights.index(default_weight) if default_weight in weights else 0))
        conf = st.slider("Confidence", 0.0, 1.0, float(CONF), 0.01)
        iou = st.slider("IOU", 0.0, 1.0, float(IOU), 0.01)
        imgsz = st.select_slider("Inference size", options=[640, 960, 1280, 1600], value=int(INFERENCE_SIZE))
        save_video = st.checkbox("Save processed video", value=False)
        output_path = st.text_input("Output path", value=OUTPUT_VIDEO, disabled=not save_video)

        st.divider()
        youtube_url = st.text_input("YouTube URL", value=DEFAULT_YOUTUBE_URL)

    if hasattr(st, "cache_resource"):
        @st.cache_resource
        def _load_model(path):
            return YOLO(path)
    else:
        def _load_model(path):
            return YOLO(path)

    def _new_heatmap(path):
        return solutions.Heatmap(
            model=path,
            classes=VEHICLE_CLASSES,
            colormap=cv2.COLORMAP_TURBO,
            conf=CONF,
            iou=IOU,
            tracker=TRACKER,
            show=False,
            show_conf=False,
            show_labels=False,
        )

    def _new_speed(path, fps):
        se = solutions.SpeedEstimator(
            model=path,
            classes=VEHICLE_CLASSES,
            conf=CONF,
            iou=IOU,
            tracker=TRACKER,
            show=False,
            fps=float(fps),
            max_hist=int(MOTION_WINDOW),
            meter_per_pixel=float(1.0 / PIXELS_PER_METER),
            max_speed=180,
        )
        _enable_live_speed_estimation(se)
        return se

    def _new_counter(path, w, h):
        count_line_y = int(h * 0.60)
        return solutions.ObjectCounter(
            model=path,
            classes=VEHICLE_CLASSES,
            conf=CONF,
            iou=IOU,
            tracker=TRACKER,
            show=False,
            show_conf=False,
            show_labels=False,
            show_in=False,
            show_out=False,
            region=[(0, count_line_y), (w, count_line_y)],
        )

    model = _load_model(weight_path)

    if "running" not in st.session_state:
        st.session_state.running = False
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = DEFAULT_YOUTUBE_URL
    if "daily_records" not in st.session_state:
        st.session_state.daily_records = {}
    if "min_stop_seconds" not in st.session_state:
        st.session_state.min_stop_seconds = 10.0
    if "reset_heatmap" not in st.session_state:
        st.session_state.reset_heatmap = False
    if "heatmap_solution" not in st.session_state:
        st.session_state.heatmap_solution = None
    if "heatmap_weight" not in st.session_state:
        st.session_state.heatmap_weight = None
    if "speed_solution" not in st.session_state:
        st.session_state.speed_solution = None
    if "count_solution" not in st.session_state:
        st.session_state.count_solution = None
    if "solutions_shared" not in st.session_state:
        st.session_state.solutions_shared = _SharedTracks()

    def _stop():
        st.session_state.running = False
        st.session_state.reset_heatmap = True

    def _start():
        st.session_state.running = True
        st.session_state.youtube_url = youtube_url.strip()

    with st.sidebar:
        st.button("Start", type="primary", on_click=_start, disabled=st.session_state.running)
        st.button("Stop", on_click=_stop, disabled=not st.session_state.running)

    live_tab, records_tab = st.tabs(["Live", "Records"])

    with live_tab:
        col1, col2 = st.columns([2, 1])
        frame_slot = col1.empty()
        with col2:
            stopped_slot = st.empty()
            moving_slot = st.empty()
            status_slot = st.empty()

    with records_tab:
        st.subheader("Daily Records")
        st.number_input(
            "Show stopped records over (seconds)",
            min_value=0.0,
            value=float(st.session_state.min_stop_seconds),
            step=1.0,
            key="min_stop_seconds",
        )
        records_summary = st.empty()
        records_table = st.empty()
        records_download = st.empty()
        records_actions = st.empty()

    def _today_key():
        return _dt.date.today().isoformat()

    def _get_daily(day_key):
        if day_key not in st.session_state.daily_records:
            st.session_state.daily_records[day_key] = {
                "moving_ids": set(),
                "stopped_seconds_by_id": defaultdict(float),
                "frames": 0,
                "moving_sum": 0,
                "stopped_sum": 0,
            }
        return st.session_state.daily_records[day_key]

    def _daily_csv(day_key):
        daily = st.session_state.daily_records.get(day_key)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["date", day_key])
        if not daily:
            w.writerow(["note", "no data"])
            return buf.getvalue()
        w.writerow(["unique_moving_vehicles", len(daily["moving_ids"])])
        w.writerow(["frames", daily["frames"]])
        if daily["frames"]:
            w.writerow(["avg_moving_per_frame", daily["moving_sum"] / daily["frames"]])
            w.writerow(["avg_stopped_per_frame", daily["stopped_sum"] / daily["frames"]])
        w.writerow([])
        w.writerow(["track_id", "stopped_seconds"])
        for track_id, seconds in sorted(daily["stopped_seconds_by_id"].items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([track_id, round(seconds, 3)])
        return buf.getvalue()

    def _render_records_view():
        day_key = _today_key()
        daily = st.session_state.daily_records.get(day_key)
        if not daily or daily["frames"] == 0:
            records_summary.info("No data recorded today yet.")
            records_table.empty()
        else:
            threshold = float(st.session_state.min_stop_seconds or 0.0)
            avg_moving = daily["moving_sum"] / max(1, daily["frames"])
            avg_stopped = daily["stopped_sum"] / max(1, daily["frames"])
            total_stopped_seconds = float(sum(daily["stopped_seconds_by_id"].values()))
            over_threshold = {
                int(track_id): float(seconds)
                for track_id, seconds in daily["stopped_seconds_by_id"].items()
                if float(seconds) >= threshold
            }

            with records_summary.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Unique moving vehicles (today)", len(daily["moving_ids"]))
                c2.metric("Avg moving / frame", f"{avg_moving:.2f}")
                c3.metric("Total stopped seconds (sum of tracks)", f"{total_stopped_seconds:.1f}")
                c4.metric(f"Stopped tracks ≥ {threshold:.0f}s", len(over_threshold))
                st.caption(f"Date: {day_key} • Frames processed: {daily['frames']} • Avg stopped / frame: {avg_stopped:.2f}")

            top_rows = [
                {"track_id": tid, "stopped_seconds": round(sec, 1)}
                for tid, sec in sorted(over_threshold.items(), key=lambda kv: kv[1], reverse=True)[:100]
            ]
            records_table.dataframe(top_rows, use_container_width=True, hide_index=True)

    def _render_records_actions():
        day_key = _today_key()
        csv_text = _daily_csv(day_key)

        records_download.download_button(
            "Download today's CSV",
            data=csv_text,
            file_name=f"traffic_records_{day_key}.csv",
            mime="text/csv",
            key=f"download_csv_{day_key}",
        )

        with records_actions.container():
            col_a, col_b = st.columns(2)
            if col_a.button("Save today's CSV to disk", use_container_width=True, key="save_csv_disk"):
                os.makedirs(RECORDS_DIR, exist_ok=True)
                path = os.path.join(RECORDS_DIR, f"traffic_records_{day_key}.csv")
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(csv_text)
                st.success(f"Saved: {path}")
            if col_b.button("Reset today's stats", use_container_width=True, key="reset_daily_stats"):
                st.session_state.daily_records.pop(day_key, None)
                st.success("Reset done.")

    _render_records_view()
    _render_records_actions()

    if (
        st.session_state.reset_heatmap
        or st.session_state.heatmap_solution is None
        or st.session_state.heatmap_weight != weight_path
    ):
        st.session_state.heatmap_solution = _new_heatmap(weight_path)
        st.session_state.speed_solution = None
        st.session_state.count_solution = None
        st.session_state.heatmap_weight = weight_path
        st.session_state.reset_heatmap = False

    if not st.session_state.running:
        status_slot.empty()
        return

    yolo_heatmap = st.session_state.heatmap_solution
    shared = st.session_state.solutions_shared

    _bind_shared_tracks(yolo_heatmap, shared)

    source = extract_youtube_stream(st.session_state.youtube_url)
    if not source:
        st.session_state.running = False
        if not source:
            st.error("Failed to extract stream. Ensure `yt-dlp` is installed and the URL is valid.")
            st.stop()

    cap = cv2.VideoCapture(source)
    writer = None

    try:
        if not cap.isOpened():
            st.error("Failed to open video source.")
            st.stop()

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_dt = 1.0 / max(1e-6, fps)

        ret, first_frame = cap.read()
        if not ret:
            st.error("Could not read the first frame.")
            st.stop()

        if save_video:
            h, w = first_frame.shape[:2]
            out_path = Path(output_path).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if st.session_state.speed_solution is None:
            st.session_state.speed_solution = _new_speed(weight_path, fps)
            _bind_shared_tracks(st.session_state.speed_solution, shared)
        if st.session_state.count_solution is None:
            h, w = first_frame.shape[:2]
            st.session_state.count_solution = _new_counter(weight_path, w, h)
            _bind_shared_tracks(st.session_state.count_solution, shared)

        speed_estimator = st.session_state.speed_solution
        object_counter = st.session_state.count_solution

        stats = {}
        processed, stopped, moving = process_frame(
            first_frame,
            model=model,
            yolo_heatmap=yolo_heatmap,
            speed_estimator=speed_estimator,
            object_counter=object_counter,
            shared_tracks=shared,
            fps=fps,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            draw_stop_moving=False,
            stats_out=stats,
        )
        if writer is not None:
            writer.write(processed)
        frame_slot.image(processed, channels="BGR", use_container_width=True)
        stopped_slot.metric("Stopped", stopped)
        moving_slot.metric("Moving", moving)

        status_slot.caption("Processing…")
        today = _get_daily(_today_key())
        today["frames"] += 1
        today["moving_sum"] += int(moving)
        today["stopped_sum"] += int(stopped)
        today["moving_ids"].update(stats.get("moving_ids", set()))
        for tid in stats.get("stopped_ids", set()):
            today["stopped_seconds_by_id"][int(tid)] += frame_dt

        while True:
            if not st.session_state.running:
                status_slot.caption("Stopped.")
                break
            ret, frame = cap.read()
            if not ret:
                status_slot.caption("Done.")
                break

            stats = {}
            processed, stopped, moving = process_frame(
                frame,
                model=model,
                yolo_heatmap=yolo_heatmap,
                speed_estimator=speed_estimator,
                object_counter=object_counter,
                shared_tracks=shared,
                fps=fps,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                draw_stop_moving=False,
                stats_out=stats,
            )
            frame_slot.image(processed, channels="BGR", use_container_width=True)
            stopped_slot.metric("Stopped", stopped)
            moving_slot.metric("Moving", moving)
            if writer is not None:
                writer.write(processed)

            day_key = _today_key()
            daily = _get_daily(day_key)
            daily["frames"] += 1
            daily["moving_sum"] += int(moving)
            daily["stopped_sum"] += int(stopped)
            daily["moving_ids"].update(stats.get("moving_ids", set()))
            for tid in stats.get("stopped_ids", set()):
                daily["stopped_seconds_by_id"][int(tid)] += frame_dt
            if daily["frames"] % 30 == 0:
                _render_records_view()
    finally:
        st.session_state.running = False
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


if __name__ == "__main__":
    if _running_in_streamlit():
        run_streamlit()
    else:
        run_cli()
