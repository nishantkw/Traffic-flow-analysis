#!/usr/bin/env python3
"""
Traffic Flow Analysis — 3-Lane Vehicle Counting with Tracking

Features
- Downloads the target YouTube video (MNn9qKG2UFI) via yt-dlp (can be skipped if --video-path provided).
- Vehicle detection using Ultralytics YOLO (pretrained COCO).
- Tracking using Ultralytics' built-in ByteTrack (fast & robust).
- Three-lane definition (default: vertical bands; or custom polygons from JSON).
- Per-lane, de-duplicated counting using virtual mid-line crossing.
- CSV logging: VehicleID, Lane, Frame, Timestamp.
- Rendered video with overlays (lane boundaries, counts).
- End-of-run summary printed to console and drawn on the last frame.

Usage
    python traffic_counter.py \
        --download \
        --ytdlp-format "bestvideo[height<=720]+bestaudio/best" \
        --model yolov8n.pt \
        --out-dir runs/exp1 \
        --fps 30 \
        --conf 0.25 \
        --device 0

Or run on a local file:
    python traffic_counter.py --video-path path/to/video.mp4 --out-dir runs/local_exp

Requirements (pip install -r requirements.txt):
    ultralytics==8.*
    yt-dlp>=2024.3.10
    opencv-python>=4.7.0.72
    numpy>=1.24
    pandas>=2.1
"""

import argparse
import csv
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# Lazy import for ultralytics (gives a clearer error if not installed)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

YOUTUBE_URL = "https://www.youtube.com/watch?v=MNn9qKG2UFI"

# --------------------- Lane utilities ---------------------

def load_lane_config(config_path: Optional[str]) -> Dict:
    """Load lane configuration JSON if provided; else return empty dict."""
    if not config_path:
        return {}
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_lane_polygons(frame_w: int, frame_h: int) -> Dict[str, List[Tuple[int, int]]]:
    """
    Fallback lane definitions: 3 vertical bands that span the frame height.
    Returns dictionary mapping lane_name -> polygon points (clockwise).
    """
    one_third = frame_w // 3
    lanes = {
        "lane_1": [(0, 0), (one_third - 1, 0), (one_third - 1, frame_h - 1), (0, frame_h - 1)],
        "lane_2": [(one_third, 0), (2 * one_third - 1, 0), (2 * one_third - 1, frame_h - 1), (one_third, frame_h - 1)],
        "lane_3": [(2 * one_third, 0), (frame_w - 1, 0), (frame_w - 1, frame_h - 1), (2 * one_third, frame_h - 1)],
    }
    return lanes


def lane_midlines(frame_w: int, frame_h: int, lane_polys: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Create a virtual horizontal count-line per lane: across the lane's bbox at 55% height.
    Returns: lane_name -> ((x1,y1), (x2,y2))
    """
    lines = {}
    for lane, poly in lane_polys.items():
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, x2 = min(xs), max(xs)
        y_top, y_bot = min(ys), max(ys)
        y = int(y_top + 0.55 * (y_bot - y_top))
        lines[lane] = ((x1, y), (x2, y))
    return lines


def point_in_poly(x: int, y: int, poly: List[Tuple[int, int]]) -> bool:
    """Ray casting to test if point (x,y) lies inside polygon poly."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


def assign_lane(cx: int, cy: int, lane_polys: Dict[str, List[Tuple[int, int]]]) -> Optional[str]:
    """Return lane name if center point lies in exactly one lane polygon, else None."""
    hits = [lane for lane, poly in lane_polys.items() if point_in_poly(cx, cy, poly)]
    if len(hits) == 1:
        return hits[0]
    return None


# --------------------- Drawing utilities ---------------------

def draw_lane_polys_and_lines(frame, lane_polys, lane_lines, lane_counts):
    for lane, poly in lane_polys.items():
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, thickness=2, color=(255, 255, 255))
        # draw midline
        (x1, y1), (x2, y2) = lane_lines[lane]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # lane label & count
        bx, by = min([p[0] for p in poly]), min([p[1] for p in poly])
        cv2.putText(frame, f"{lane}  count={lane_counts.get(lane,0)}", (bx + 5, by + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2, cv2.LINE_AA)


def draw_track(frame, tid, bbox, label):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(frame, f"{label} ID:{tid}", (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)


# --------------------- CSV logger ---------------------

class CsvLogger:
    def __init__(self, csv_path: str, fps: float):
        self.csv_path = csv_path
        self.fps = fps
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["vehicle_id", "lane", "frame", "timestamp"])

    def log(self, vehicle_id: int, lane: str, frame_idx: int):
        ts = timedelta(seconds=frame_idx / max(self.fps, 1e-9))
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([vehicle_id, lane, frame_idx, str(ts)])


# --------------------- Download util ---------------------

def download_youtube(url: str, out_path: str, ytdlp_format: str) -> str:
    """
    Download a YouTube video using yt-dlp.
    Returns the downloaded file path.
    """
    import subprocess
    os.makedirs(out_path, exist_ok=True)
    # note: ensure yt-dlp is installed
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", ytdlp_format,
        "-o", os.path.join(out_path, "%(title).80s.%(ext)s"),
        url
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Find the newest file in out_path
    files = sorted([os.path.join(out_path, f) for f in os.listdir(out_path)], key=lambda p: os.path.getmtime(p))
    if not files:
        raise RuntimeError("yt-dlp did not produce any file.")
    return files[-1]


# --------------------- Main pipeline ---------------------

def main():
    ap = argparse.ArgumentParser(description="3-Lane Traffic Counter with YOLO + ByteTrack")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--download", action="store_true", help="Download the target YouTube video")
    src.add_argument("--video-path", type=str, default=None, help="Path to a local video file")

    ap.add_argument("--youtube-url", type=str, default=YOUTUBE_URL, help="YouTube URL to download")
    ap.add_argument("--ytdlp-format", type=str, default="bestvideo[height<=720]+bestaudio/best", help="yt-dlp -f FORMAT")
    ap.add_argument("--out-dir", type=str, default="runs/exp", help="Output directory")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model (e.g., yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--device", type=str, default=None, help="Computation device, e.g., '0' or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    ap.add_argument("--fps", type=float, default=None, help="Override FPS; else read from video")
    ap.add_argument("--lane-config", type=str, default=None, help="Path to JSON with lane polygons")
    ap.add_argument("--classes", type=int, nargs='*', default=[2, 3, 5, 7], help="COCO class IDs to keep (2,3,5,7 are car, motorcycle, bus, truck)")
    ap.add_argument("--show", action="store_true", help="Show preview window")
    ap.add_argument("--save", action="store_true", help="Save annotated video")
    ap.add_argument("--skip-download-if-exists", action="store_true", help="If download target already exists, skip.")
    args = ap.parse_args()

    if YOLO is None:
        print("ERROR: 'ultralytics' is not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Acquire video
    if args.video_path:
        video_path = args.video_path
    elif args.download:
        dl_dir = out_dir / "downloads"
        if args.skip_download_if_exists and dl_dir.exists() and any(dl_dir.iterdir()):
            # reuse newest file
            video_path = sorted(dl_dir.iterdir(), key=lambda p: p.stat().st_mtime)[-1].as_posix()
            print(f"Reusing existing download: {video_path}")
        else:
            video_path = download_youtube(args.youtube_url, dl_dir.as_posix(), args.ytdlp_format)
            print(f"Downloaded to: {video_path}")
    else:
        print("ERROR: Provide --video-path or use --download.", file=sys.stderr)
        sys.exit(2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(3)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps or src_fps
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2) Lanes
    cfg = load_lane_config(args.lane_config)
    if cfg and "lanes" in cfg:
        lane_polys = {k: [(int(x), int(y)) for x, y in v] for k, v in cfg["lanes"].items()}
    else:
        lane_polys = default_lane_polygons(frame_w, frame_h)
    lane_lines = lane_midlines(frame_w, frame_h, lane_polys)

    # 3) CSV logger
    csv_logger = CsvLogger((out_dir / "counts.csv").as_posix(), fps=fps)

    # 4) YOLO + tracking config
    model = YOLO(args.model)  # will auto-download if needed

    # 5) Video writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter((out_dir / "annotated.mp4").as_posix(), fourcc, fps, (frame_w, frame_h))

    # 6) State for per-lane, per-ID crossing
    counted = {lane: set() for lane in lane_polys.keys()}
    last_positions: Dict[int, Tuple[int, int]] = {}

    # 7) Process loop using Ultralytics streaming track API
    # Filter: vehicle classes (COCO ids)
    names = None
    frame_idx = -1
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run tracking for this single frame (persist=True keeps tracks alive)
        # ByteTrack is the default tracker in Ultralytics if 'tracker' not passed.
        results = model.track(
            source=frame,
            stream=True,
            persist=True,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
            verbose=False,
        )

        # 'results' yields exactly one item here because source=frame
        res = next(results)
        if names is None:
            names = res.names

        # Draw lanes
        draw_lane_polys_and_lines(frame, lane_polys, lane_lines, {k: len(v) for k, v in counted.items()})

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            ids = None
            if hasattr(res.boxes, "id") and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)

            for i, bbox in enumerate(boxes_xyxy):
                cls_id = clss[i]
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                if ids is None:
                    # If ByteTrack didn't return an ID (rare), skip counting to avoid duplicates
                    continue
                tid = int(ids[i])

                # center point
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                lane = assign_lane(cx, cy, lane_polys)
                if lane:
                    draw_track(frame, tid, bbox, label)

                    # crossing logic: if last y above the lane line and now below (or vice versa), count once
                    (lx1, ly), (lx2, _) = lane_lines[lane]
                    prev = last_positions.get(tid)
                    if prev is not None:
                        _, py = prev
                        crossed = (py < ly and cy >= ly) or (py > ly and cy <= ly)
                        if crossed and tid not in counted[lane]:
                            counted[lane].add(tid)
                            csv_logger.log(vehicle_id=tid, lane=lane, frame_idx=frame_idx)
                    last_positions[tid] = (cx, cy)

        # Show/save
        overlay = frame.copy()
        # total banner
        totals = " | ".join([f"{ln}:{len(s)}" for ln, s in counted.items()])
        cv2.rectangle(overlay, (0, 0), (frame_w, 40), (0, 0, 0), -1)
        cv2.putText(overlay, f"Counts -> {totals}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        frame = overlay

        if args.show:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # Summary
    print("\n=== Summary ===")
    for lane, ids in counted.items():
        print(f"{lane}: {len(ids)} vehicles")
    print(f"CSV: {(out_dir / 'counts.csv').as_posix()}")
    if args.save:
        print(f"Annotated video: {(out_dir / 'annotated.mp4').as_posix()}")


if __name__ == "__main__":
    main()
