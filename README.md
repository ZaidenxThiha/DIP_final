# SOME APPLICATIONS OF COMPUTER VISION USING YOLO FOR FINAL REPORT

Reference: https://docs.ultralytics.com/solutions/

Task:
Analyze at least 03 among the list of applications in the reference link.

This project implements and analyzes **four** Ultralytics “Solutions” applications (ex1–ex4). Each exercise is runnable from the repo root and uses local paths that resolve correctly by default.

## Project Structure

- `ex1/`: heatmap + tracking + speed estimation analytics on a local video (on people)
- `ex2/`: traffic heatmap from YouTube (CLI + Streamlit UI + records)(on vehicles)
- `ex3/`: parking occupancy (spot selection + parking manager)
- `ex4/`: instance segmentation on a local video
- `models/`: weights used by the demos (`yolo11n.pt`, `yolo11n-seg.pt`)

## Installation

Core dependencies:

```bash
python3 -m pip install ultralytics opencv-python numpy
```

For `ex2` Streamlit mode and YouTube stream extraction:

```bash
python3 -m pip install streamlit yt-dlp
```

Notes:

- `ex2` extracts an HLS (`m3u8`) stream via `yt-dlp --js-runtime node`, so Node.js is typically required (`node` on PATH).

---

## Application 1: Human Activity Detection (ex1)

Script: `ex1/ex1.py`

What it does:

- Runs YOLO detection + multi-object tracking once per frame and accumulates movement into a **trajectory heatmap** (Ultralytics `solutions.Heatmap`).
- Adds simple analytics: **left/right occupancy** (by midline) and **approx speed** per track.
- Shows a `STOP` label when the estimated speed drops below a threshold.

Inputs/outputs:

- Input video (default): `ex1/ex1.mp4`
- Output video (default): `ex1/final_output_heatmap.mp4`

Run:

```bash
python3 ex1/ex1.py
```

Key parameters (why they matter):

- `CONF`, `IOU`, `TRACKER`: affect detection quality and track stability; unstable IDs lead to noisy heatmaps and wrong speed estimates.
- `PIXELS_PER_METER`, `MOTION_WINDOW`: convert pixel motion into an approximate speed; these are not calibrated unless you measure scene scale.
- `--stop-speed`: sets the “stopped” threshold in m/s; too high will mislabel slow motion as stopped.

Strengths / limitations:

- Strength: heatmaps quickly reveal **where** people spend time and common paths.
- Limitation: “speed” is approximate and depends heavily on camera angle and the pixel-to-meter scale.

---

## Application 2: Traffic Analytics (Heatmap + Speed + Counting) (ex2)

Script: `ex2/ex2.py`

What it does:

- Uses Ultralytics Solutions to render: **Heatmap + SpeedEstimator + ObjectCounter** on a YouTube stream.
- Classifies tracks as **stopped vs moving** based on SpeedEstimator output.
- Overlays `STOP ID:<track_id>` on stopped vehicles; if local heat is high, also overlays `JAM <heat>`.
- In Streamlit mode, records **stopped seconds by track_id** and allows CSV export.

Inputs/outputs:

- Input: YouTube URL (stream extracted via `yt-dlp`)
- Output video (default): `ex2/traffic_analysis_heatmap.mp4`
- Records folder (Streamlit “Save to disk”): `ex2/records/`

Run (CLI / OpenCV window):

```bash
python3 ex2/ex2.py
```

Run (Streamlit UI):

```bash
streamlit run ex2/ex2.py
```

Key parameters (why they matter):

- `VEHICLE_CLASSES`: which objects are tracked (COCO IDs); wrong classes produce irrelevant counts and heatmaps.
- `STOP_SPEED_THRESHOLD`: determines stopped vs moving; sensitive to jitter in tracking and camera perspective.
- `CONGESTION_THRESHOLD`, `HEATMAP_DECAY`: control “jam” sensitivity and how long heat persists.

Strengths / limitations:

- Strength: combines multiple analytics layers (where vehicles accumulate, how fast they move, and how many pass a line).
- Limitation: stop detection from vision alone can misfire in occlusions or when the tracker ID switches.

---

## Application 3: Parking Management (ex3)

Scripts:

- `ex3/bounding_boxes.py`: interactive parking-spot selection (generates `ex3/bounding_boxes.json`)
- `ex3/parking.py`: runs Ultralytics `solutions.ParkingManagement` to mark spots as occupied/free

What it does:

- You define parking bays once (polygons) using a selection tool.
- The ParkingManagement solution tracks vehicles and decides whether each bay is occupied based on overlap and tracking.

Inputs/outputs:

- Input video (default): `ex3/parking_video.mp4`
- Spot definition: `ex3/bounding_boxes.json`
- Optional output video: `-o ex3/parking_output.mp4`

Run:

```bash
python3 ex3/bounding_boxes.py
python3 ex3/parking.py
```

Key parameters (why they matter):

- `json_file`: quality of bay polygons determines accuracy; bad polygons lead to false occupancy.
- `classes=[2]` (car): restricts detection to the vehicle class; include more classes if you expect trucks/buses.
- `conf`, `iou`, `tracker`: influence tracking stability; instability causes bay flicker (occupied/free toggling).

Strengths / limitations:

- Strength: practical occupancy estimation with minimal setup (one-time spot annotation).
- Limitation: strong dependence on camera angle and occlusions; long occlusions can break tracking IDs.

---

## Application 4: Instance Segmentation (ex4)

Script: `ex4/instance_segmentation.py`

What it does:

- Runs Ultralytics `solutions.InstanceSegmentation` to produce per-object masks (e.g., cars) and overlays them on video frames.

Inputs/outputs:

- Input video (default): `ex4/car_video.mp4`
- Optional output video: `-o ex4/instance_segmentation_output.mp4`

Run:

```bash
python3 ex4/instance_segmentation.py
```

Key parameters (why they matter):

- `--conf`, `--iou`: tradeoff between missing objects vs including false positives.
- Segmentation weights (`models/yolo11n-seg.pt`): segmentation models are required; detection-only weights won’t output masks.

Strengths / limitations:

- Strength: masks enable pixel-accurate reasoning (e.g., lane occupancy, area estimation) beyond bounding boxes.
- Limitation: segmentation is heavier than detection; FPS is typically lower on CPU.
