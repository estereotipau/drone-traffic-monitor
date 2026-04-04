#!/usr/bin/env python3
"""
DJI Mini 4 Pro - Post-flight processing pipeline.

Takes a 4K video + SQLite database from a flight and produces:
- High-resolution crops of each detected vehicle
- Deduplicated dataset (no repeated vehicles)
- Improved color classification
- Clean CSV/SQLite output

Usage:
    python process_flight.py --video flight.mp4 --db detections.db --output output_dir/
    python process_flight.py --video flight.mp4 --db detections.db --model yolov8m --skip-dedup
"""

import argparse
import csv
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    frame_idx: int
    timestamp_sec: float
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    crop: np.ndarray = field(repr=False, default=None)
    color_name: str = ""
    color_hsv: tuple = (0, 0, 0)
    embedding: np.ndarray = field(repr=False, default=None)
    track_id: int = -1
    is_duplicate: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Post-flight vehicle detection pipeline")
    parser.add_argument("--video", required=True, help="Path to 4K video file")
    parser.add_argument("--db", required=False, help="Path to detections.db (for telemetry correlation)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="yolov8s", choices=["yolov8n", "yolov8s", "yolov8m"],
                        help="YOLO model size (default: yolov8s)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--frame-interval", type=int, default=5,
                        help="Process every N-th frame (default: 5)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication step")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for dedup (default: 0.85)")
    parser.add_argument("--min-crop-size", type=int, default=30,
                        help="Minimum crop dimension in pixels (default: 30)")
    parser.add_argument("--flight-id", type=int, default=None,
                        help="Flight ID to correlate with in the database")
    parser.add_argument("--roi-top", type=float, default=0.0,
                        help="ROI top boundary as fraction of frame height (default: 0.0 = no ROI)")
    parser.add_argument("--roi-bottom", type=float, default=1.0,
                        help="ROI bottom boundary as fraction of frame height (default: 1.0 = no ROI)")
    parser.add_argument("--sahi", action="store_true",
                        help="Use SAHI (Slicing Aided Hyper Inference) for small object detection")
    parser.add_argument("--sahi-slice-size", type=int, default=640,
                        help="SAHI slice size in pixels (default: 640)")
    parser.add_argument("--sahi-overlap", type=float, default=0.2,
                        help="SAHI overlap ratio between slices (default: 0.2)")
    return parser.parse_args()


# ── YOLO Detection ──

def load_yolo(model_name: str):
    from ultralytics import YOLO
    print(f"[1/5] Loading {model_name}...")
    model = YOLO(f"{model_name}.pt")
    return model


def detect_in_video(model, video_path: str, frame_interval: int, conf: float,
                    min_crop_size: int, roi_top: float = 0.0, roi_bottom: float = 1.0,
                    use_sahi: bool = False, sahi_slice_size: int = 640,
                    sahi_overlap: float = 0.2) -> list[DetectionResult]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_y1 = int(height * roi_top)
    roi_y2 = int(height * roi_bottom)
    has_roi = roi_top > 0.0 or roi_bottom < 1.0

    print(f"[2/5] Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    if has_roi:
        print(f"      ROI: rows {roi_y1}-{roi_y2} ({roi_top:.0%}-{roi_bottom:.0%} of frame)")
    if use_sahi:
        print(f"      SAHI enabled: slice={sahi_slice_size}px, overlap={sahi_overlap:.0%}")
    print(f"      Analyzing every {frame_interval}th frame ({total_frames // frame_interval} frames to process)")

    # Setup SAHI if requested
    sahi_model = None
    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=model.ckpt_path,
                confidence_threshold=conf,
            )
            print(f"      SAHI model loaded")
        except ImportError:
            print("      WARNING: sahi not installed (pip install sahi), falling back to standard detection")
            use_sahi = False

    detections = []
    frame_idx = 0
    processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Apply ROI crop
            if has_roi:
                roi_frame = frame[roi_y1:roi_y2, :]
            else:
                roi_frame = frame

            if use_sahi and sahi_model:
                from sahi.predict import get_sliced_prediction
                sahi_result = get_sliced_prediction(
                    roi_frame, sahi_model,
                    slice_height=sahi_slice_size, slice_width=sahi_slice_size,
                    overlap_height_ratio=sahi_overlap, overlap_width_ratio=sahi_overlap,
                    verbose=0,
                )
                for pred in sahi_result.object_prediction_list:
                    bbox = pred.bbox
                    x1, y1_r = int(bbox.minx), int(bbox.miny)
                    x2, y2_r = int(bbox.maxx), int(bbox.maxy)
                    # Offset back to full frame coords
                    y1 = y1_r + roi_y1
                    y2 = y2_r + roi_y1
                    w, h = x2 - x1, y2 - y1
                    if w < min_crop_size or h < min_crop_size:
                        continue
                    crop = frame[y1:y2, x1:x2].copy()
                    detections.append(DetectionResult(
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / fps,
                        class_id=pred.category.id,
                        class_name=pred.category.name,
                        confidence=pred.score.value,
                        bbox=(x1, y1, x2, y2),
                        crop=crop,
                    ))
            else:
                results = model(roi_frame, conf=conf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1_r, x2, y2_r = box.xyxy[0].cpu().numpy().astype(int)
                        # Offset back to full frame coords
                        y1 = y1_r + roi_y1
                        y2 = y2_r + roi_y1
                        w, h = x2 - x1, y2 - y1
                        if w < min_crop_size or h < min_crop_size:
                            continue

                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        confidence = float(box.conf[0])

                        crop = frame[y1:y2, x1:x2].copy()

                        detections.append(DetectionResult(
                            frame_idx=frame_idx,
                            timestamp_sec=frame_idx / fps,
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            crop=crop,
                        ))

            processed += 1
            if processed % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / processed * (total_frames // frame_interval - processed)
                print(f"      Frame {frame_idx}/{total_frames} | {len(detections)} detections | ETA: {eta:.0f}s")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"      Done: {len(detections)} detections in {elapsed:.1f}s ({processed} frames processed)")
    return detections


# ── Color Classification ──

COLOR_RANGES = [
    # (name, h_min, h_max, s_min, v_min)
    ("red", 0, 10, 0.15, 0.2),
    ("orange", 10, 25, 0.15, 0.2),
    ("yellow", 25, 40, 0.15, 0.2),
    ("green", 40, 85, 0.15, 0.2),
    ("cyan", 85, 100, 0.15, 0.2),
    ("blue", 100, 130, 0.15, 0.2),
    ("purple", 130, 155, 0.15, 0.2),
    ("pink", 155, 170, 0.15, 0.2),
    ("red", 170, 180, 0.15, 0.2),  # red wraps around
]


def classify_color(crop: np.ndarray) -> tuple[str, tuple]:
    """Classify dominant color of a crop using HSV histogram analysis."""
    h, w = crop.shape[:2]
    # Sample center 60%
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    center = crop[margin_y:h - margin_y, margin_x:w - margin_x]
    if center.size == 0:
        center = crop

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()  # 0-179 in OpenCV
    s_mean = hsv[:, :, 1].mean() / 255.0  # normalize to 0-1
    v_mean = hsv[:, :, 2].mean() / 255.0

    # Achromatic
    if v_mean < 0.15:
        return "black", (h_mean * 2, s_mean, v_mean)
    if s_mean < 0.12 and v_mean > 0.65:
        return "white", (h_mean * 2, s_mean, v_mean)
    if s_mean < 0.12:
        return "silver" if v_mean > 0.4 else "gray", (h_mean * 2, s_mean, v_mean)

    # Chromatic — h_mean is 0-179 in OpenCV, convert to 0-360 logic
    h_deg = h_mean  # OpenCV uses 0-179
    for name, h_min, h_max, s_min, v_min in COLOR_RANGES:
        if h_min <= h_deg < h_max and s_mean >= s_min and v_mean >= v_min:
            return name, (h_mean * 2, s_mean, v_mean)

    return "unknown", (h_mean * 2, s_mean, v_mean)


def classify_colors(detections: list[DetectionResult]) -> list[DetectionResult]:
    print(f"[3/5] Classifying colors for {len(detections)} detections...")
    for det in detections:
        if det.crop is not None:
            det.color_name, det.color_hsv = classify_color(det.crop)
    return detections


# ── Deduplication ──

def compute_embeddings(detections: list[DetectionResult]) -> list[DetectionResult]:
    """Compute visual embeddings for each crop using a lightweight CNN."""
    try:
        import torchvision.transforms as T
        import torch
        import torchvision.models as models
    except ImportError:
        print("      WARNING: torch/torchvision not available, skipping embedding-based dedup")
        return detections

    print(f"[4/5] Computing embeddings for {len(detections)} detections...")

    # Use ResNet18 as feature extractor
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # remove classification head
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for i, det in enumerate(detections):
            if det.crop is not None:
                img = cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB)
                tensor = transform(img).unsqueeze(0)
                embedding = model(tensor).squeeze().numpy()
                # L2 normalize
                norm = np.linalg.norm(embedding)
                det.embedding = embedding / norm if norm > 0 else embedding

            if (i + 1) % 100 == 0:
                print(f"      {i + 1}/{len(detections)} embeddings computed")

    return detections


def deduplicate(detections: list[DetectionResult], threshold: float) -> list[DetectionResult]:
    """Remove duplicate detections using cosine similarity of embeddings."""
    if not any(d.embedding is not None for d in detections):
        print("      No embeddings available, skipping dedup")
        return detections

    print(f"      Deduplicating with threshold {threshold}...")

    # Group by class for efficiency
    by_class = {}
    for det in detections:
        by_class.setdefault(det.class_name, []).append(det)

    unique = []
    total_dupes = 0

    for cls_name, cls_dets in by_class.items():
        cls_unique = []
        for det in cls_dets:
            if det.embedding is None:
                cls_unique.append(det)
                continue

            is_dupe = False
            for existing in cls_unique:
                if existing.embedding is None:
                    continue
                sim = np.dot(det.embedding, existing.embedding)
                if sim > threshold:
                    # Keep the one with higher confidence
                    if det.confidence > existing.confidence:
                        cls_unique.remove(existing)
                        cls_unique.append(det)
                    is_dupe = True
                    total_dupes += 1
                    break

            if not is_dupe:
                cls_unique.append(det)

        unique.extend(cls_unique)

    print(f"      Removed {total_dupes} duplicates: {len(detections)} → {len(unique)}")

    # Tag duplicates (instead of discarding)
    unique_set = set(id(d) for d in unique)
    for det in detections:
        det.is_duplicate = id(det) not in unique_set

    return unique


# ── Telemetry Correlation ──

def load_telemetry(db_path: str, flight_id: int = None) -> list[dict]:
    """Load telemetry data from SQLite to correlate with detections."""
    if not db_path or not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM detections"
    params = ()
    if flight_id:
        query += " WHERE flight_id = ?"
        params = (flight_id,)
    query += " ORDER BY timestamp"

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def correlate_telemetry(detections: list[DetectionResult], telemetry: list[dict]) -> list[DetectionResult]:
    """Attach telemetry data to detections by closest timestamp."""
    if not telemetry:
        return detections

    # Build timestamp index from telemetry
    telem_times = []
    for t in telemetry:
        try:
            ts = int(t.get("timestamp", 0)) / 1000.0  # ms to seconds
            telem_times.append((ts, t))
        except (ValueError, TypeError):
            pass

    if not telem_times:
        return detections

    telem_times.sort(key=lambda x: x[0])
    base_time = telem_times[0][0]

    print(f"      Correlating {len(detections)} detections with {len(telem_times)} telemetry records")

    for det in detections:
        # Find closest telemetry record by timestamp
        det_time = base_time + det.timestamp_sec
        best_telem = min(telem_times, key=lambda x: abs(x[0] - det_time))
        t = best_telem[1]
        det.track_id = t.get("detection_id", -1)

    return detections


# ── Output ──

def save_all_crops(detections: list[DetectionResult], output_dir: str, db_path: str = None,
                   flight_id: int = None):
    """Save ALL crops and raw database before deduplication."""
    print(f"[3/5] Saving all {len(detections)} crops and raw DB...")

    crops_dir = os.path.join(output_dir, "crops_hd")
    os.makedirs(crops_dir, exist_ok=True)

    telemetry = load_telemetry(db_path, flight_id) if db_path else []

    # Save all crops
    for i, det in enumerate(detections):
        crop_filename = f"{i:05d}_{det.class_name}_{det.color_name}.jpg"
        det.crop_filename = crop_filename  # store for later reference
        crop_path = os.path.join(crops_dir, crop_filename)
        if det.crop is not None:
            cv2.imwrite(crop_path, det.crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Save raw DB (all detections, before dedup)
    raw_db_path = os.path.join(output_dir, "dataset_raw.db")
    conn = sqlite3.connect(raw_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY,
            frame_idx INTEGER,
            timestamp_sec REAL,
            class_name TEXT,
            confidence REAL,
            color_name TEXT,
            color_h REAL, color_s REAL, color_v REAL,
            bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
            crop_width INTEGER, crop_height INTEGER,
            crop_path TEXT,
            drone_lat REAL, drone_lng REAL, drone_alt REAL,
            gimbal_pitch REAL, gimbal_yaw REAL, gimbal_roll REAL
        )
    """)

    for i, det in enumerate(detections):
        crop_h, crop_w = (det.crop.shape[:2] if det.crop is not None else (0, 0))
        lat, lng, alt, g_pitch, g_yaw, g_roll = _get_telemetry(det, telemetry)
        conn.execute("""
            INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (i, det.frame_idx, det.timestamp_sec, det.class_name, det.confidence,
              det.color_name, det.color_hsv[0], det.color_hsv[1], det.color_hsv[2],
              det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
              crop_w, crop_h, getattr(det, 'crop_filename', ''),
              lat, lng, alt, g_pitch, g_yaw, g_roll))

    conn.commit()
    conn.close()
    print(f"      Saved: {raw_db_path} ({len(detections)} records)")
    print(f"      Saved: {len(detections)} crops to {crops_dir}/")


def move_duplicate_crops(detections: list[DetectionResult], output_dir: str):
    """Move duplicate crops to a subfolder."""
    crops_dir = os.path.join(output_dir, "crops_hd")
    dupes_dir = os.path.join(crops_dir, "duplicates")
    os.makedirs(dupes_dir, exist_ok=True)

    moved = 0
    for det in detections:
        if det.is_duplicate and hasattr(det, 'crop_filename'):
            src = os.path.join(crops_dir, det.crop_filename)
            dst = os.path.join(dupes_dir, det.crop_filename)
            if os.path.exists(src):
                os.rename(src, dst)
                moved += 1

    print(f"      Moved {moved} duplicate crops to {dupes_dir}/")


def _get_telemetry(det, telemetry):
    """Extract telemetry for a detection."""
    lat, lng, alt = 0.0, 0.0, 0.0
    g_pitch, g_yaw, g_roll = 0.0, 0.0, 0.0
    if telemetry:
        closest = min(telemetry,
                      key=lambda t: abs(t.get("frame_number", 99999) - det.frame_idx),
                      default=None)
        if closest:
            lat = closest.get("drone_lat", 0.0)
            lng = closest.get("drone_lng", 0.0)
            alt = closest.get("drone_alt", 0.0)
            g_pitch = closest.get("gimbal_pitch", 0.0)
            g_yaw = closest.get("gimbal_yaw", 0.0)
            g_roll = closest.get("gimbal_roll", 0.0)
    return lat, lng, alt, g_pitch, g_yaw, g_roll


def save_results(detections: list[DetectionResult], output_dir: str, db_path: str = None,
                 flight_id: int = None):
    """Save deduplicated results."""
    print(f"[5/5] Saving {len(detections)} unique results to {output_dir}/")

    telemetry = load_telemetry(db_path, flight_id) if db_path else []

    # Build CSV with unique detections only
    csv_path = os.path.join(output_dir, "dataset.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "frame_idx", "timestamp_sec", "class_name", "confidence",
            "color_name", "color_h", "color_s", "color_v",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "crop_width", "crop_height", "crop_path",
            "drone_lat", "drone_lng", "drone_alt",
            "gimbal_pitch", "gimbal_yaw", "gimbal_roll",
        ])

        for i, det in enumerate(detections):
            crop_filename = getattr(det, 'crop_filename', f"{i:05d}_{det.class_name}_{det.color_name}.jpg")
            lat, lng, alt, g_pitch, g_yaw, g_roll = _get_telemetry(det, telemetry)
            crop_h, crop_w = (det.crop.shape[:2] if det.crop is not None else (0, 0))

            writer.writerow([
                i, det.frame_idx, f"{det.timestamp_sec:.2f}",
                det.class_name, f"{det.confidence:.3f}",
                det.color_name, f"{det.color_hsv[0]:.1f}", f"{det.color_hsv[1]:.3f}", f"{det.color_hsv[2]:.3f}",
                det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                crop_w, crop_h, crop_filename,
                lat, lng, alt,
                g_pitch, g_yaw, g_roll,
            ])

    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    class_counts = {}
    color_counts = {}
    for d in detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        color_counts[d.color_name] = color_counts.get(d.color_name, 0) + 1

    with open(summary_path, "w") as f:
        f.write(f"Total unique detections: {len(detections)}\n\n")
        f.write("By class:\n")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {cls}: {cnt}\n")
        f.write("\nBy color:\n")
        for col, cnt in sorted(color_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {col}: {cnt}\n")

    # Also create a SQLite output
    out_db_path = os.path.join(output_dir, "dataset.db")
    conn = sqlite3.connect(out_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY,
            frame_idx INTEGER,
            timestamp_sec REAL,
            class_name TEXT,
            confidence REAL,
            color_name TEXT,
            color_h REAL, color_s REAL, color_v REAL,
            bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
            crop_width INTEGER, crop_height INTEGER,
            crop_path TEXT,
            drone_lat REAL, drone_lng REAL, drone_alt REAL,
            gimbal_pitch REAL, gimbal_yaw REAL, gimbal_roll REAL
        )
    """)

    for i, det in enumerate(detections):
        crop_filename = getattr(det, 'crop_filename', f"{i:05d}_{det.class_name}_{det.color_name}.jpg")
        crop_h, crop_w = (det.crop.shape[:2] if det.crop is not None else (0, 0))
        lat, lng, alt, g_pitch, g_yaw, g_roll = _get_telemetry(det, telemetry)

        conn.execute("""
            INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (i, det.frame_idx, det.timestamp_sec, det.class_name, det.confidence,
              det.color_name, det.color_hsv[0], det.color_hsv[1], det.color_hsv[2],
              det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
              crop_w, crop_h, crop_filename,
              lat, lng, alt, g_pitch, g_yaw, g_roll))

    conn.commit()
    conn.close()

    print(f"      Saved: {csv_path}")
    print(f"      Saved: {out_db_path}")
    print(f"      Saved: {len(detections)} unique vehicles")
    print(f"      Summary: {summary_path}")


# ── Main ──

def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # 1. Load YOLO and run detection
    model = load_yolo(args.model)
    detections = detect_in_video(model, args.video, args.frame_interval, args.conf, args.min_crop_size,
                                roi_top=args.roi_top, roi_bottom=args.roi_bottom,
                                use_sahi=args.sahi, sahi_slice_size=args.sahi_slice_size,
                                sahi_overlap=args.sahi_overlap)

    if not detections:
        print("No detections found. Exiting.")
        sys.exit(0)

    # 2. Classify colors
    detections = classify_colors(detections)

    # 3. Save ALL crops and raw DB (before dedup)
    save_all_crops(detections, args.output, args.db, args.flight_id)

    # 4. Deduplicate
    if not args.skip_dedup:
        all_detections = detections  # keep reference to all
        detections = compute_embeddings(detections)
        unique = deduplicate(detections, args.dedup_threshold)
        # Move duplicate crops to subfolder
        move_duplicate_crops(all_detections, args.output)
    else:
        print("[4/5] Skipping deduplication (--skip-dedup)")
        unique = detections

    # 5. Save deduplicated results
    save_results(unique, args.output, args.db, args.flight_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
