# drone-traffic-monitor

Real-time vehicle detection and tracking from a DJI Mini 4 Pro drone, with on-device YOLO inference, live web dashboard, and a post-flight processing pipeline for building traffic datasets.

Built as a pedagogical tool for the **Economics degree program (Licenciatura en Economia) at Universidad de Buenos Aires (UBA)**, to teach students about real-world data collection, computer vision pipelines, and statistical dataset construction.

## What it does

1. **On-device detection**: YOLOv8 runs directly on a Samsung S23+ connected to the drone via DJI MSDK V5. Vehicles are detected, tracked, and classified in real time.
2. **Vehicle tracking**: A centroid-based tracker deduplicates vehicles across frames, ensuring each vehicle is counted once even when the drone is stationary.
3. **Live dashboard**: A web dashboard (served from the phone) shows telemetry, detection stats, and a 7x7 grid of recent vehicle crops in real time.
4. **Data persistence**: Every unique vehicle is logged to SQLite with class, color, confidence, bounding box, GPS coordinates, altitude, gimbal angle, and a cropped image.
5. **Post-flight pipeline**: A Python pipeline re-processes the 4K video from the drone's SD card, producing high-resolution crops, embedding-based deduplication, and a clean dataset.

## Architecture

```
DJI Mini 4 Pro (4K camera)
    |
    | OcuSync / DJI O4
    v
DJI RC-NS (remote controller)
    |
    | USB-C
    v
Samsung S23+ (Kotlin app)
    +-- MSDK V5: video feed, telemetry, camera control
    +-- YOLOv8 (ONNX Runtime): vehicle detection
    +-- Vehicle tracker: centroid-based dedup
    +-- SQLite: detection + telemetry persistence
    +-- NanoHTTPD: live web dashboard (:8080)
    +-- Crop storage: JPG per vehicle
    |
    | WiFi (same network or Tailscale)
    v
Notebook browser: http://<phone-ip>:8080
    +-- Real-time telemetry
    +-- Detection feed (7x7 crop grid)
    +-- Flight stats

Post-flight:
    4K video (SD card) + detections.db
        |
        v
    Python pipeline (process_flight.py)
        +-- YOLOv8s/m on 4K frames
        +-- ResNet18 embeddings for dedup
        +-- Color classification
        +-- ROI filtering
        +-- Optional SAHI for small objects
        |
        v
    Clean dataset: HD crops + CSV + SQLite
```

## Features

### Android App (Kotlin + MSDK V5)

| Feature | Description |
|---------|-------------|
| **Video feed** | Live camera stream via `ICameraStreamManager` |
| **YOLO detection** | On-device YOLOv8n inference (ONNX Runtime), toggleable |
| **Dual models** | Switch between VisDrone (aerial-optimized, 10 classes) and COCO (80 classes) |
| **Vehicle tracker** | Centroid-based tracking, saves best crop per vehicle |
| **ROI** | Configurable detection zone (skip top/bottom of frame) |
| **Telemetry** | Battery, altitude, distance, speed, GPS, gimbal angle, satellite count |
| **Photo capture** | Via UI button or RC trigger (right joystick button) |
| **4K recording** | Optional toggle to record 4K to SD card simultaneously |
| **Dashboard** | Embedded web server with SSE + polling fallback |
| **Return to Home** | RTH button with SDK integration |
| **Color extraction** | Dominant color classification per vehicle (HSV-based) |
| **SQLite logging** | Full detection metadata + drone telemetry per vehicle |

### Post-flight Pipeline (Python)

```bash
python pipeline/process_flight.py \
    --video DJI_0001.MP4 \
    --db detections.db \
    --output results/ \
    --model yolov8s \
    --roi-top 0.15 --roi-bottom 0.80
```

| Feature | Description |
|---------|-------------|
| **YOLO models** | `yolov8n` (fast), `yolov8s` (balanced), `yolov8m` (accurate) |
| **ROI** | Configurable detection zone via `--roi-top` / `--roi-bottom` |
| **SAHI** | Slicing Aided Hyper Inference for small objects (`--sahi`) |
| **Deduplication** | ResNet18 embeddings + cosine similarity |
| **Raw preservation** | `dataset_raw.db` with all detections; duplicates moved to subfolder |
| **Color classification** | HSV-based dominant color extraction |
| **Telemetry correlation** | Matches detections with drone GPS/altitude/gimbal from SQLite |
| **Output** | `dataset.csv` + `dataset.db` + HD crops + summary |

## Hardware Requirements

| Component | Role |
|-----------|------|
| **DJI Mini 4 Pro** | Drone (sub-249g, consumer) |
| **DJI RC-NS** | Remote controller (supports third-party apps via USB) |
| **Android phone** (Snapdragon 8 Gen 2+) | Runs the MSDK app + YOLO inference |
| **PC/Laptop** | Development, dashboard viewing, post-flight processing |

The DJI RC-N2 also works. The DJI RC 2 (with built-in screen) does **not** support MSDK apps.

## Setup

### Prerequisites

- JDK 17
- Android SDK (platform 35, build-tools 35.0.0, NDK 21.4.7075529)
- [DJI Developer account](https://developer.dji.com/) and App Key
- Python 3.10+ with `ultralytics`, `opencv-python`, `torch`, `torchvision`
- ADB and scrcpy for development

### Build the Android App

```bash
cd app/

# Copy and edit configuration
cp gradle.properties.example gradle.properties
cp local.properties.example local.properties
# Edit both files with your paths and DJI App Key

# Download YOLO models to assets/
cd app/src/main/assets/
python3 -c "
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='onnx', imgsz=640, simplify=True)
# For VisDrone model:
import urllib.request
urllib.request.urlretrieve(
    'https://huggingface.co/mshamrai/yolov8n-visdrone/resolve/main/best.pt',
    'yolov8n_visdrone.pt'
)
YOLO('yolov8n_visdrone.pt').export(format='onnx', imgsz=640, simplify=True)
"
cd ../../../..

# Build
./gradlew assembleDebug

# Install (via ADB over WiFi)
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Connect to Drone

1. Power on the DJI Mini 4 Pro and RC-NS
2. Connect phone to RC-NS via USB-C
3. Grant USB permission when prompted
4. The app initializes the SDK, connects to the drone, and starts the video feed

### Access Dashboard

From any device on the same network:
```
http://<phone-ip>:8080
```

Or via ADB port forwarding:
```bash
adb forward tcp:8080 tcp:8080
# then open http://localhost:8080
```

## MSDK V5 Gotchas

If you're building your own DJI MSDK V5 app, here are the non-obvious issues we encountered:

1. **`Helper.install(this)`** must be called in `Application.attachBaseContext()`, not `onCreate()`. Without it, all `dji.v5.manager.*` classes throw `NoClassDefFoundError`.

2. **Don't reference DJI classes in the Application class body.** The classloader resolves them before `attachBaseContext` runs. Keep DJI imports in Activity classes only.

3. **Use `cameraStreamManager`**, not `videoStreamManager`. The latter is deprecated and returns `null` for newer drones like Mini 4 Pro.

4. **`AvailableCameraUpdatedListener`** has two abstract methods. Missing `onCameraStreamEnableUpdate` causes `AbstractMethodError` at runtime (no compile-time warning).

5. **The video stream is always 1080p.** No API can change this — it's a bandwidth limitation of the RC-to-aircraft link. 4K is only available on the SD card.

6. **`ComponentIndexType.LEFT_OR_MAIN`** is the correct camera index for Mini 4 Pro.

## Regulatory Note

The DJI Mini 4 Pro is a sub-249g drone classified as a "very light" UAV. In Argentina (ANAC regulations), this class can be operated in urban areas without special permits, provided the operator maintains visual line of sight and stays away from airports and restricted zones.

## License

MIT

## Acknowledgments

- [DJI Mobile SDK V5](https://github.com/dji-sdk/Mobile-SDK-Android-V5)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [NanoHTTPD](https://github.com/NanoHttpd/nanohttpd)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
