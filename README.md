# Behaviour Analysis  - time in zone

Using Computer vision and techniques from Supervision librairy, I developed a solution  to track customer movements in supermarkets to generate 2D heatmaps
for behavior analysis.






https://github.com/user-attachments/assets/2bdb3f02-bdd1-4a9c-8f5d-81d54f2df80a



## 💻 install

- clone repository

- setup python environment and activate it [optional]

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- install required dependencies

  ```bash
  pip install -r requirements.txt
  ```

## 🛠 scripts

### `download_from_youtube`

This script allows you to download a video from YouTube.

- `--url`: The full URL of the YouTube video you wish to download.
- `--output_path` (optional): Specifies the directory where the video will be saved.
- `--file_name` (optional): Sets the name of the saved video file.

```bash
python scripts/download_from_youtube.py \
--url "https://www.youtube.com/watch?v=-8zyEwAa50Q" \
--output_path "data/checkout" \
--file_name "video.mp4"
```

```bash
python scripts/download_from_youtube.py \
--url "https://www.youtube.com/watch?v=MNn9qKG2UFI" \
--output_path "data/traffic" \
--file_name "video.mp4"
```

### `stream_from_file`

This script allows you to stream video files from a directory. It's an awesome way to
mock a live video stream for local testing. Video will be streamed in a loop under
`rtsp://localhost:8554/live0.stream` URL. This script requires docker to be installed.

- `--video_directory`: Directory containing video files to stream.
- `--number_of_streams`: Number of video files to stream.

```bash
python scripts/stream_from_file.py \
--video_directory "data/checkout" \
--number_of_streams 1
```

```bash
python scripts/stream_from_file.py \
--video_directory "data/traffic" \
--number_of_streams 1
```

### `draw_zones`

If you want to test zone time in zone analysis on your own video, you can use this
script to design custom zones and save results as a JSON file. The script will open a
window where you can draw polygons on the source image or video file. The polygons will
be saved as a JSON file.

- `--source_path`: Path to the source image or video file for drawing polygons.
- `--zone_configuration_path`: Path where the polygon annotations will be saved as a JSON file.


- `enter` - finish drawing the current polygon.
- `escape` - cancel drawing the current polygon.
- `q` - quit the drawing window.
- `s` - save zone configuration to a JSON file.

```bash
python scripts/draw_zones.py \
--source_path "data/checkout/video.mp4" \
--zone_configuration_path "data/checkout/config.json"
```

```bash
python scripts/draw_zones.py \
--source_path "data/traffic/video.mp4" \
--zone_configuration_path "data/traffic/config.json"
```


## 🎬 video & stream processing



### `main`

Script to run object detection on a video file using the Ultralytics YOLOv8 model.

  - `--zone_configuration_path`: Path to the zone configuration JSON file.
  - `--source_video_path`: Path to the source video file.
  - `--target_video_path` : Path to the target video file (output)
  - `--target_heatmap_video_path` : Path to the target heatmap video file (output)
  - `--dataheatmap_background` : Path to the background image for the heatmap
  - `--weights`: Path to the model weights file. Default is `'yolov8s.pt'`.
  - `--device`: Computation device (`'cpu'`, `'mps'` or `'cuda'`). Default is `'cpu'`.
  - `--classes`: List of class IDs to track. If empty, all classes are tracked.
  - `--confidence_threshold`: Confidence level for detections (`0` to `1`). Default is `0.3`.
  - `--iou_threshold`: IOU threshold for non-max suppression. Default is `0.7`.



## © license

This is a an extension to an existing code from Roboflow Supervision Tutorial. This demo integrates two main components, each with its own licensing:

- ultralytics: The object detection model used in this demo, YOLOv8, is distributed
  under the [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  You can find more details about this license here.

- supervision: The analytics code that powers the zone-based analysis in this demo is
  based on the Supervision library, which is licensed under the
  [MIT license](https://github.com/roboflow/supervision/blob/develop/LICENSE.md). This
  makes the Supervision part of the code fully open source and freely usable in your
  projects.
