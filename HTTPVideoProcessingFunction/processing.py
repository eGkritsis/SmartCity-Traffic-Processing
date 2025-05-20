import cv2
import numpy as np
from datetime import datetime
import logging
import os
from collections import deque

# Constants
DASHED_LINE_DISTANCE = 20  # meters between dashed lines
MIN_SPEED_THRESHOLD = 5    # km/h - ignore vehicles moving slower than this
MAX_SPEED_THRESHOLD = 150  # km/h - cap maximum reasonable speed to 150 km/h
PIXELS_PER_METER = 5       # Approximate pixel to meter conversion
SPEED_SMOOTHING_WINDOW = 3 # Number of speed samples to average

class VehicleTracker:
    def __init__(self):
        self.tracked_vehicles = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections, frame_time, frame_height):
        updated_vehicles = {}
        results = []
        
        # Mark all tracked vehicles as unmatched initially
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            vehicle['matched'] = False
        
        for detection in detections:
            x, y, w, h, confidence, class_id = detection
            center_x = x + w // 2
            center_y = y + h // 2
            
            matched_id = None
            min_distance = float('inf')
            
            for vehicle_id, vehicle in self.tracked_vehicles.items():
                if vehicle['matched']:
                    continue
                    
                last_x, last_y, last_time = vehicle['last_position']
                distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                time_elapsed = frame_time - last_time
                
                if time_elapsed < 1.0 and distance < 100:  # 1 sec max, 100 pixels max
                    if distance < min_distance:
                        min_distance = distance
                        matched_id = vehicle_id
            
            if matched_id is not None:
                vehicle = self.tracked_vehicles[matched_id]
                vehicle['matched'] = True
                
                if len(vehicle['positions']) > 0:
                    prev_x, prev_y, prev_time = vehicle['positions'][-1]
                    pixels_moved = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    time_elapsed = frame_time - prev_time
                    
                    if time_elapsed > 0:
                        meters_moved = pixels_moved / PIXELS_PER_METER
                        speed_mps = meters_moved / time_elapsed
                        speed_kmh = speed_mps * 3.6
                        
                        if MIN_SPEED_THRESHOLD < speed_kmh < MAX_SPEED_THRESHOLD:
                            vehicle['speed_history'].append(speed_kmh)
                            # Keep only last N speeds for smoothing
                            if len(vehicle['speed_history']) > SPEED_SMOOTHING_WINDOW:
                                vehicle['speed_history'].pop(0)
                            # Smoothed speed
                            vehicle['speed'] = sum(vehicle['speed_history']) / len(vehicle['speed_history'])

                            # REAL TIME ALERTS PRINTED TO THE LOG CONSOLE
                            if vehicle['speed'] > 130:
                                logging.warning(
                                    f"[ALERT] Vehicle ID {matched_id} exceeded 130 km/h: "
                                    f"{vehicle['speed']:.1f} km/h at time {frame_time:.2f}s, position={center_x},{center_y}"
                                )
                
                vehicle['last_position'] = (center_x, center_y, frame_time)
                vehicle['positions'].append((center_x, center_y, frame_time))
                updated_vehicles[matched_id] = vehicle
                
                # Only report vehicles with enough positions
                if 'speed' in vehicle and len(vehicle['positions']) >= 5:
                    # Determine lane by dominant movement vector
                    dx = vehicle['positions'][-1][0] - vehicle['positions'][0][0]
                    dy = vehicle['positions'][-1][1] - vehicle['positions'][0][1]
                    if abs(dx) > abs(dy):
                        lane = 'inbound' if dx < 0 else 'outbound'
                    else:
                        lane = 'inbound' if dy < 0 else 'outbound'
                    
                    results.append({
                        'id': matched_id,
                        'type': 'car' if class_id in [2, 3] else 'truck',
                        'lane': lane,
                        'speed': vehicle['speed'],
                        'timestamp': frame_time,
                        'position': (center_x, center_y),
                        'confidence': confidence
                    })
            else:
                # New vehicle
                vehicle_id = self.next_id
                self.next_id += 1
                
                updated_vehicles[vehicle_id] = {
                    'id': vehicle_id,
                    'type': 'car' if class_id in [2, 3] else 'truck',
                    'last_position': (center_x, center_y, frame_time),
                    'positions': [(center_x, center_y, frame_time)],
                    'speed_history': [],
                    'first_seen': frame_time,
                    'matched': True,
                    'confidence': confidence
                }
        
        # Remove vehicles not seen for more than 2 seconds
        for vehicle_id, vehicle in self.tracked_vehicles.items():
            if not vehicle['matched']:
                last_seen = frame_time - vehicle['last_position'][2]
                if last_seen < 2.0:
                    updated_vehicles[vehicle_id] = vehicle
        
        self.tracked_vehicles = updated_vehicles
        self.frame_count += 1
        return results

def process_video_clip(video_path: str) -> dict:
    logging.info(f"Starting enhanced video processing: {video_path}")
    
    base_dir = os.path.dirname(__file__)
    net = cv2.dnn.readNet(
        os.path.join(base_dir, "yolov3-tiny.weights"),
        os.path.join(base_dir, "yolov3-tiny.cfg")
    )
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    logging.info(
        f"Video properties - {width}x{height} @ {fps:.1f}fps, "
        f"Duration: {duration:.1f}s, Frames: {total_frames}"
    )

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

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / fps

        # Vehicle detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        detections = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.8 and class_id in [2, 3, 5, 7]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detections.append((x, y, w, h, confidence, class_id))

        tracked_vehicles = tracker.update(detections, current_time, height)
        results["vehicles"].extend(tracked_vehicles)

        if frame_count % int(fps * 10) == 0:
            logging.info(
                f"Frame {frame_count}/{total_frames} ({current_time:.1f}s) - "
                f"Tracking {len(tracker.tracked_vehicles)} vehicles"
            )

    cap.release()
    
    results["processing_end"] = datetime.utcnow().isoformat()
    processing_time = (datetime.utcnow() - datetime.fromisoformat(results["processing_start"])).total_seconds()
    
    logging.info(
        f"Processing completed. Detected {len(results['vehicles'])} vehicle tracks "
        f"in {processing_time:.1f} seconds ({processing_time/duration:.1f}x realtime)"
    )
    
    return results
