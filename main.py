import argparse
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

import matplotlib.pyplot as plt

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)
HEAT_MAP_ANNOTATOR = sv.HeatMapAnnotator()

class HeatMap_Visualizer:
    def __init__( self,
        position = sv.Position.BOTTOM_CENTER,
        opacity: float = 0.2,
        radius: int = 20,
        kernel_size: int = 25,
        top_hue: int = 0,
        low_hue: int = 125,
    ):
    
        """
        Args:
            position (Position): The position of the heatmap. Defaults to
                `BOTTOM_CENTER`.
            opacity (float): Opacity of the overlay mask, between 0 and 1.
            radius (int): Radius of the heat circle.
            kernel_size (int): Kernel size for blurring the heatmap.
            top_hue (int): Hue at the top of the heatmap. Defaults to 0 (red).
            low_hue (int): Hue at the bottom of the heatmap. Defaults to 125 (blue).
        """
        self.position = position
        self.opacity = opacity
        self.radius = radius
        self.kernel_size = kernel_size
        self.top_hue = top_hue
        self.low_hue = low_hue
        self.heat_mask = None

    def visualize(self, scene, points):
        """
        Args:
        scene (ImageType): The image where the heatmap will be drawn.
            `ImageType` is a flexible type, accepting either `numpy.ndarray`
            or `PIL.Image.Image`.
        points (List[Tuple[int, int]]): List of points to draw the heatmap on.

    Returns:
        The heatmap image, matching the type of `scene` (`numpy.ndarray`
            or `PIL.Image.Image`)
        """
        assert isinstance(scene, np.ndarray)
        if self.heat_mask is None:
            self.heat_mask = np.zeros(scene.shape[:2], dtype=np.float32)

        mask = np.zeros(scene.shape[:2])
        for xy in points:
            x, y = int(xy[0]), int(xy[1])
            cv2.circle(
                img=mask,
                center=(x, y),
                radius=self.radius,
                color=(1,),
                thickness=-1,  # fill
            )
        self.heat_mask = mask + self.heat_mask
        temp = self.heat_mask.copy()
        temp = self.low_hue - temp / temp.max() * (self.low_hue - self.top_hue)
        temp = temp.astype(np.uint8)
        if self.kernel_size is not None:
            temp = cv2.blur(temp, (self.kernel_size, self.kernel_size))
        hsv = np.zeros(scene.shape)
        hsv[..., 0] = temp
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(self.heat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) > 0
        scene[mask] = cv2.addWeighted(temp, self.opacity, scene, 1 - self.opacity, 0)[
            mask
        ]
        return scene
        
HEAT_MAP_VISUALIZER = HeatMap_Visualizer()

SOURCE = np.array([[156, 67], [327, 216], [154, 348], [21, 140]])

TARGET_WIDTH = 450
TARGET_HEIGHT = 450

TARGET = np.array(
    [
        [0, 300],
        [300, 300],
        [300, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

def main(
    source_video_path: str,
    zone_configuration_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)

    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.BOTTOM_CENTER,),
        )
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]
    # Initialize heatmap canvas
    heatmap_scene = np.zeros((750, 750, 3), dtype=np.uint8)
    heatmap_bg = cv2.imread(args.dataheatmap_background)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        with sv.VideoSink(args.target_heatmap_video_path, sv.VideoInfo(width=750, height=750, fps=video_info.fps, total_frames=video_info.total_frames)) as hm_sink:
            for frame in frames_generator:
                results = model(frame, verbose=False, device=device, conf=confidence)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[find_in_list(detections.class_id, classes)]
                detections = detections.with_nms(threshold=iou)
                detections = tracker.update_with_detections(detections)

                annotated_frame = frame.copy()

                for idx, zone in enumerate(zones):
                    annotated_frame = sv.draw_polygon(
                        scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                    )

                    detections_in_zone = detections#[zone.trigger(detections)]
                    time_in_zone = timers[idx].tick(detections_in_zone)
                    custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                    new_points = detections_in_zone.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )
                    transformed_points = view_transformer.transform_points(points=new_points).astype(int)

                    heatmap_scene = HEAT_MAP_VISUALIZER.visualize(heatmap_scene, transformed_points)


                    annotated_frame = COLOR_ANNOTATOR.annotate(
                        scene=annotated_frame,
                        detections=detections_in_zone,
                        custom_color_lookup=custom_color_lookup,
                    )
                    labels = [
                        f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                        for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                    ]
                    annotated_frame = LABEL_ANNOTATOR.annotate(
                        scene=annotated_frame,
                        detections=detections_in_zone,
                        labels=labels,
                        custom_color_lookup=custom_color_lookup,
                    )

                    annotated_frame = HEAT_MAP_ANNOTATOR.annotate(
                        scene=annotated_frame,
                        detections=detections_in_zone
                    )

                sink.write_frame(annotated_frame)
                cv2.imshow("Processed Video", annotated_frame)

                # draw heatmap scene above the background where heatmap is not zero in all channels
                heatmap_image = heatmap_bg.copy()
                mask = np.any(heatmap_scene > 0, axis=-1)
                heatmap_image[mask] = heatmap_scene.transpose(0, 1, 2)[mask]

                hm_sink.write_frame(heatmap_image)
                cv2.imshow("HeatMap Video", heatmap_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--target_heatmap_video_path",
        required=True,
        help="Path to the target heatmap video file (output)",
        type=str,
    )
    parser.add_argument(
        "--dataheatmap_background",
        required=True,
        help="Path to the background image for the heatmap",
        type=str,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8s.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
