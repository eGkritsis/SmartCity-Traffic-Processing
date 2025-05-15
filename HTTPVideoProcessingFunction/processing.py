import os
import cv2
import numpy as np
from datetime import datetime
import logging

# Distance between dashed lines (in meters)
DASHED_LINE_DISTANCE = 20  

def process_video_clip(video_path: str) -> dict:
    """Process a video clip to detect vehicles and calculate statistics."""
    logging.info(f"Starting video processing: {video_path}")
    
    # Load YOLO config and weights using absolute paths
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "yolov3-tiny.cfg")
    weights_path = os.path.join(base_dir, "yolov3-tiny.weights")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"YOLO config file not found: {cfg_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logging.info(f"Video properties - FPS: {fps}, Frames: {total_frames}, Duration: {duration:.2f}s")

    # Load YOLO network
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    vehicles = {}
    results = {
        "vehicles": [],
        "frame_stats": [],
        "processing_start": datetime.utcnow().isoformat()
    }

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # Prepare image for detection
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id in [2, 3, 5, 7]:  # Vehicle types: car, truck, bus, etc.
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                vehicle_id = f"{class_ids[i]}_{x}_{y}"
                lane = "inbound" if y < height // 2 else "outbound"
                vehicle_type = "car" if class_ids[i] in [2, 3] else "truck"

                if vehicle_id in vehicles:
                    prev_x, prev_y, prev_time = vehicles[vehicle_id]
                    pixels_moved = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    time_elapsed = current_time - prev_time

                    if time_elapsed > 0:
                        meters_moved = pixels_moved * (DASHED_LINE_DISTANCE / 100)  # Simplified estimate
                        speed_mps = meters_moved / time_elapsed
                        speed_kmh = speed_mps * 3.6

                        results["vehicles"].append({
                            "id": vehicle_id,
                            "type": vehicle_type,
                            "lane": lane,
                            "speed": speed_kmh,
                            "timestamp": current_time,
                            "position": (x, y)
                        })

                        if speed_kmh > 130:
                            logging.warning(
                                f"High speed alert! {vehicle_type} at {speed_kmh:.1f} km/h "
                                f"(Lane: {lane}, Time: {current_time:.1f}s)"
                            )

                vehicles[vehicle_id] = (x, y, current_time)

        if frame_count % int(fps * 60) == 0:
            logging.info(f"Processed {frame_count}/{total_frames} frames ({current_time:.1f}s)")

    cap.release()
    results["processing_end"] = datetime.utcnow().isoformat()
    results["total_frames"] = frame_count
    results["duration"] = duration
    logging.info(f"Video processing completed. Detected {len(results['vehicles'])} vehicles")

    return results
