import cv2
import numpy as np
from datetime import datetime
import logging
import os
from collections import defaultdict

# ---------- Logging Configuration ----------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Prevent duplicate handlers
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('traffic_processing.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# ---------- Constants ----------
DASHED_LINE_DISTANCE = 20          # meters between dashed lines
MIN_SPEED_THRESHOLD = 5            # km/h - minimum speed to consider
MAX_SPEED_THRESHOLD = 200          # km/h - maximum plausible speed
PIXELS_DASHED_LINE = 245
# pixel to meter conversion factor
PIXELS_PER_METER = PIXELS_DASHED_LINE / DASHED_LINE_DISTANCE 
SPEED_SMOOTHING_WINDOW = 3         # number of frames to average speed over
CONFIDENCE = 0.6                  # confidence threshold for car/truck detection

# ---------- Vehicle Tracker Class ----------
class VehicleTracker:
    def __init__(self):
        """Initialize vehicle tracker with empty structures"""
        self.tracked_vehicles = {}  # Active vehicles being tracked
        self.available_ids = set()   # Reusable vehicle IDs
        self.next_id = 1             # Next available new ID
        self.alerted_vehicles = {}   # Track last alert time per vehicle
        self.max_speeds = {}         # Track maximum speed per vehicle

    def update(self, detections, frame_time, frame_height):
        """
        Update vehicle tracking with new detections
        Args:
            detections: List of (x,y,w,h,confidence,class_id)
            frame_time: Current video time in seconds
            frame_height: Video frame height for position scaling
        Returns:
            List of vehicle results with max speeds
        """
        updated_vehicles = {}
        results = []

        # Mark all existing vehicles as unmatched at start of update
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            vehicle['matched'] = False

        for x, y, w, h, conf, class_id in detections:
            cx, cy = x + w // 2, y + h // 2

            matched_id = None
            min_dist = float('inf')
            for vid, v in self.tracked_vehicles.items():
                if v['matched']:
                    continue
                lx, ly, lt = v['last_position']
                dist = np.hypot(cx - lx, cy - ly)
                if (frame_time - lt < 1.0) and dist < 100 and dist < min_dist:
                    matched_id = vid
                    min_dist = dist

            if matched_id is not None:
                v = self.tracked_vehicles[matched_id]
                v['matched'] = True
                if v['positions']:
                    px, py, pt = v['positions'][-1]
                    pixels = np.hypot(cx - px, cy - py)
                    t_elapsed = frame_time - pt
                    if t_elapsed > 0:
                        meters = pixels / PIXELS_PER_METER
                        speed_kmh = (meters / t_elapsed) * 3.6
                        if MIN_SPEED_THRESHOLD < speed_kmh < MAX_SPEED_THRESHOLD:
                            v['speed_history'].append(speed_kmh)
                            if len(v['speed_history']) > SPEED_SMOOTHING_WINDOW:
                                v['speed_history'].pop(0)
                            v['speed'] = sum(v['speed_history']) / len(v['speed_history'])

                            if v['speed'] > 130 and not v.get('alerted', False):
                                logger.warning(
                                    f"[SPEED ALERT] Vehicle ID {matched_id} ({v['type']}) exceeded 130 km/h: "
                                    f"{v['speed']:.1f} km/h at time {frame_time:.2f}s, position=({cx},{cy})"
                                )
                                v['alerted'] = True  # Mark alerted

                v['last_position'] = (cx, cy, frame_time)
                v['positions'].append((cx, cy, frame_time))
                updated_vehicles[matched_id] = v

                if 'speed' in v and len(v['positions']) >= 3:
                    dx = v['positions'][-1][0] - v['positions'][0][0]
                    dy = v['positions'][-1][1] - v['positions'][0][1]
                    direction = (
                        'inbound'
                        if (abs(dx) > abs(dy) and dx < 0) or (abs(dy) >= abs(dx) and dy < 0)
                        else 'outbound'
                    )
                    results.append({
                        'id': matched_id,
                        'type': v['type'],
                        'direction': direction,
                        'speed': v['speed'],
                        'timestamp': frame_time,
                        'position': (cx, cy),
                        'confidence': conf
                    })

            else:
                # Always assign a new unique ID without reuse
                new_id = self.next_id
                self.next_id += 1
                updated_vehicles[new_id] = {
                    'id': new_id,
                    'type': 'car' if class_id == 2 else 'truck',
                    'last_position': (cx, cy, frame_time),
                    'positions': [(cx, cy, frame_time)],
                    'speed_history': [],
                    'matched': True,
                    'alerted': False,
                    'confidence': conf
                }

        # Keep unmatched vehicles only if updated recently (3 seconds)
        for vid, v in self.tracked_vehicles.items():
            if not v['matched'] and frame_time - v['last_position'][2] < 3.0:
                updated_vehicles[vid] = v

        self.tracked_vehicles = updated_vehicles
        return results


# ---------- Video Processing Function ----------
def process_video_clip(video_path: str) -> dict:
    """Process a video clip and return vehicle tracking results"""
    logger.info(f"Starting video processing: {video_path}")

    # Load YOLOv3-tiny model
    base_dir = os.path.dirname(__file__)
    net = cv2.dnn.readNet(
        os.path.join(base_dir, "yolov3-tiny.weights"),
        os.path.join(base_dir, "yolov3-tiny.cfg")
    )
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    logger.info(f"Video properties - {width}x{height} @ {fps:.1f}fps, Duration: {duration:.1f}s, Frames: {total_frames}")

    # Initialize tracker and results structure
    tracker = VehicleTracker()
    results = {
        "vehicles": [],
        "processing_start": datetime.utcnow().isoformat(),
        "video_properties": {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "resolution": f"{width}x{height}"
        }
    }

    # Process each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # Detect vehicles using YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process detections (only cars and trucks)
        detections = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE and class_id in [2, 7]:  # Only cars (2) and trucks (7)
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    detections.append((x, y, w, h, confidence, class_id))

        # Update tracker with new detections
        tracked_vehicles = tracker.update(detections, current_time, height)
        results["vehicles"].extend(tracked_vehicles)

        # Log progress every 10 seconds
        if frame_count % int(fps * 10) == 0:
            logger.info(f"Progress: {frame_count}/{total_frames} frames ({current_time:.1f}/{duration:.1f}s)")

    # Finalize results
    cap.release()
    results["processing_end"] = datetime.utcnow().isoformat()
    proc_time = (datetime.utcnow() - datetime.fromisoformat(results["processing_start"])).total_seconds()
    
    # Filter to only keep max speed per vehicle
    unique_vehicles = {}
    for vehicle in results["vehicles"]:
        vid = vehicle['id']
        if vid not in unique_vehicles or vehicle['speed'] > unique_vehicles[vid]['speed']:
            unique_vehicles[vid] = vehicle
    
    results["vehicles"] = list(unique_vehicles.values())
    
    logger.info(f"Processing completed. {len(results['vehicles'])} unique vehicles in {proc_time:.1f}s")
    return results