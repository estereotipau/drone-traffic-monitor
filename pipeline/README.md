# DJI Mini 4 Pro - Post-flight Processing Pipeline

## Quick Start

```bash
python process_flight.py \
    --video /path/to/DJI_0001.MP4 \
    --db /path/to/detections.db \
    --output /path/to/output/ \
    --model yolov8s
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | (required) | Path to 4K video from drone SD card |
| `--db` | (optional) | Path to detections.db from the app (for telemetry) |
| `--output` | (required) | Output directory |
| `--model` | `yolov8s` | YOLO model: `yolov8n` (fast), `yolov8s` (balanced), `yolov8m` (accurate) |
| `--conf` | 0.35 | Confidence threshold |
| `--frame-interval` | 5 | Process every N-th frame (lower = more detections, slower) |
| `--skip-dedup` | false | Skip embedding-based deduplication |
| `--dedup-threshold` | 0.85 | Cosine similarity threshold for dedup (higher = stricter) |
| `--min-crop-size` | 30 | Minimum crop dimension in pixels |
| `--flight-id` | (optional) | Flight ID to correlate in the database |

## Output

```
output/
├── crops_hd/           # High-res vehicle crops (from 4K frames)
│   ├── 00001_car_white.jpg
│   ├── 00002_van_blue.jpg
│   └── ...
├── dataset.csv         # Full dataset in CSV format
├── dataset.db          # Full dataset in SQLite format
└── summary.txt         # Detection/color statistics
```

## Examples

Fast pass (nano model, no dedup):
```bash
python process_flight.py --video DJI_0001.MP4 --output out/ --model yolov8n --skip-dedup
```

Full pipeline with strict dedup:
```bash
python process_flight.py --video DJI_0001.MP4 --db detections.db --output out/ --model yolov8m --dedup-threshold 0.90
```

Dense sampling (every 2 frames):
```bash
python process_flight.py --video DJI_0001.MP4 --output out/ --frame-interval 2
```
