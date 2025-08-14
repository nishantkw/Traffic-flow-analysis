# Traffic Flow Analysis — 3-Lane Vehicle Counter

This project provides a **complete, fast, and robust** traffic flow analysis pipeline that:
- Detects vehicles using **YOLO (Ultralytics)** pre-trained on COCO
- Tracks vehicles across frames using **ByteTrack** (via Ultralytics tracking API)
- Defines **three distinct lanes** (default vertical bands; or custom polygons)
- **Counts per lane** with de-duplication (crossing-based)
- Exports **CSV** and renders an **annotated video** with live lane counts
- Prints a **summary** at the end

**Target video**: https://www.youtube.com/watch?v=MNn9qKG2UFI

---

## 1. Quick Start

```bash
git clone <your-repo-url>
cd traffic_flow_analysis

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Option A — Auto-download the YouTube video
```bash
python traffic_counter.py --download --out-dir runs/exp1 --save --show
```
- Uses `yt-dlp` to download the video at up to 720p.
- Produces:
  - `runs/exp1/annotated.mp4` — overlayed output
  - `runs/exp1/counts.csv` — `vehicle_id,lane,frame,timestamp`

### Option B — Use a local file
```bash
python traffic_counter.py --video-path /path/to/video.mp4 --out-dir runs/local_exp --save
```

**Recommended flags:**
- `--model yolov8n.pt` (default) for real-time or near real-time on standard hardware
- `--conf 0.25` detection confidence (adjust for precision/recall)
- `--imgsz 640` for speed/accuracy balance
- `--device 0` to run on GPU 0 if available

---

## 2. Lane Definition

By default, the script splits the frame into **three vertical lanes**. For precise control, supply a JSON file using polygon coordinates (clockwise). See [`lane_config_example.json`](lane_config_example.json).

**Example JSON:**
```json
{
  "lanes": {
    "lane_1": [[50, 50], [600, 50], [600, 700], [50, 700]],
    "lane_2": [[610, 50], [1160, 50], [1160, 700], [610, 700]],
    "lane_3": [[1170, 50], [1720, 50], [1720, 700], [1170, 700]]
  }
}
```

Pass it with:
```bash
python traffic_counter.py --download --lane-config lane_config_example.json --save --show
```

> Tip: Adjust these polygons to the actual lanes visible in your chosen camera angle. Polygons may overlap slightly at borders; a detection is assigned when its **center point** lies inside exactly one polygon.

---

## 3. How Counting Works

- We compute a **virtual horizontal mid-line** inside each lane polygon.
- For each tracked vehicle **ID**, we store its previous center. When it **crosses the lane's mid-line**, we increment that lane's unique count **once** for that ID.
- We filter to COCO classes `[2,3,5,7]` → `car, motorcycle, bus, truck` by default (override via `--classes`).

CSV columns:
- `vehicle_id`: Tracker ID (stable across frames)
- `lane`: Lane name (e.g., `lane_2`)
- `frame`: Frame index where the count was registered
- `timestamp`: Time since video start (derived from FPS)

---

## 4. Performance Notes

- Default model `yolov8n.pt` is **tiny** and runs fast on CPU/GPU.
- For higher accuracy, try `yolov8s.pt` or bigger, at the cost of speed.
- Downscale with `--imgsz 640` (or 480) for more FPS.
- If your machine lacks a GPU, keep the resolution modest and `--conf` a bit higher to reduce false positives.

---

## 5. Demo Video (1–2 minutes)

After running with `--save`, trim the resulting `annotated.mp4` into a short demo:
```bash
# Example: extract a 90-second clip starting at 00:05
ffmpeg -ss 00:00:05 -i runs/exp1/annotated.mp4 -t 00:01:30 -c copy runs/exp1/demo_90s.mp4
```
Upload the demo to Google Drive or similar and include the link in your submission.

---

## 6. Repository Structure

```
traffic_flow_analysis/
├─ traffic_counter.py
├─ requirements.txt
├─ lane_config_example.json
├─ README.md
└─ runs/                # auto-created outputs (csv, annotated video, downloads)
```

---

## 7. Troubleshooting

- **ultralytics not found** → `pip install ultralytics`
- **yt-dlp not found** → `pip install yt-dlp`
- **Video opens but no detections** → decrease `--conf`, try `--model yolov8s.pt`
- **Counts seem off** → adjust lane polygons; ensure mid-line crosses vehicle flow; raise `--conf`
- **Slow** → reduce `--imgsz`, close other apps, try GPU `--device 0`

---

## 8. Technical Summary

**Approach.** We use **YOLO (Ultralytics)** with COCO weights for vehicle detection and **ByteTrack** (Ultralytics tracker) for ID persistence. Lanes are polygons; assignment uses **point-in-polygon** test at the detection center. We count per-lane when a track **crosses** an internal horizontal **mid-line**—this prevents duplication.

**Challenges & Solutions.**
- *Lane assignment ambiguity at borders* → require the center point to be in **exactly one** polygon.
- *Duplicate counts due to jitter* → only increment when crossing the mid-line and remember counted IDs per lane.
- *Performance vs. accuracy* → default to `yolov8n.pt` with `imgsz=640`, with easy switches for bigger models.

**Accuracy.** YOLOv8 COCO weights are strong general detectors. ByteTrack maintains identities well without the overhead of appearance embeddings (like DeepSORT), supporting near real-time on standard hardware.

---

## 9. Citation / Attributions

- Ultralytics YOLO: https://docs.ultralytics.com
- ByteTrack: https://arxiv.org/abs/2110.06864
- Video (for testing): https://www.youtube.com/watch?v=MNn9qKG2UFI
