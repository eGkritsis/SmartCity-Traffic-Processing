import cv2
import numpy as np
import logging

# Constants (from assignment)
FRAME_RATE = 25  # fps
DISTANCE_BETWEEN_LANES_M = 20  # meters between green lines, used as scale
SPEED_LIMIT_CAR_KMH = 90
SPEED_LIMIT_TRUCK_KMH = 80
REALTIME_ALERT_SPEED_KMH = 130

# Assuming detection class (simplified)
class Vehicle:
    def __init__(self, id, bbox, vehicle_type='car'):
        self.id = id
        self.bbox = bbox  # bounding box (x,y,w,h)
        self.vehicle_type = vehicle_type
        self.positions = []  # to track centroid over frames
        self.speed_kmh = 0
        self.counted = False  # if counted for stats

def estimate_speed(distance_m, time_s):
    # Speed = distance / time (m/s) * 3.6 -> km/h
    if time_s == 0:
        return 0
    return (distance_m / time_s) * 3.6

def process_video_clip(filepath):
    logging.info(f"Starting processing of video clip: {filepath}")

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    # Background subtractor for moving object detection
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    vehicle_id_counter = 0
    vehicles = {}
    vehicle_speeds = []
    speed_violations_count = 0
    high_speed_alerts = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fg_mask = back_sub.apply(frame)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        # Find contours (moving objects)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # filter out small noise
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detected_bboxes.append((x, y, w, h))

        # Simple tracking logic (centroid overlap)
        # This is a naive tracker for demo purposes
        current_centroids = [(int(x + w/2), int(y + h/2)) for (x,y,w,h) in detected_bboxes]

        # Match detected vehicles to existing by closest centroid
        for centroid, bbox in zip(current_centroids, detected_bboxes):
            matched = False
            for v_id, vehicle in vehicles.items():
                last_pos = vehicle.positions[-1] if vehicle.positions else None
                if last_pos:
                    dist = np.linalg.norm(np.array(centroid) - np.array(last_pos))
                    if dist < 50:  # threshold for matching
                        vehicle.positions.append(centroid)
                        vehicle.bbox = bbox
                        matched = True
                        break
            if not matched:
                # New vehicle detected
                vehicle_id_counter += 1
                # For simplicity, classify all as 'car' here; extend as needed
                vehicles[vehicle_id_counter] = Vehicle(vehicle_id_counter, bbox, vehicle_type='car')
                vehicles[vehicle_id_counter].positions.append(centroid)

        # Calculate speed if possible
        for vehicle in vehicles.values():
            if len(vehicle.positions) >= 2:
                # Distance travelled in pixels between last two positions
                p1 = vehicle.positions[-2]
                p2 = vehicle.positions[-1]
                pixel_distance = np.linalg.norm(np.array(p2) - np.array(p1))

                # Estimate speed assuming 20m between lanes corresponds to certain pixel distance
                # TODO: calibrate pixel_to_meter scale - for now assume pixel_distance ~ meters (simplified)
                # Here we mock pixel_to_meter scale as 1 pixel = 0.1 meters (example)
                pixel_to_meter = 0.1
                distance_m = pixel_distance * pixel_to_meter

                # Time between frames = 1/FRAME_RATE
                time_s = 1 / FRAME_RATE

                vehicle.speed_kmh = estimate_speed(distance_m, time_s)

                # Speed limit checks
                speed_limit = SPEED_LIMIT_CAR_KMH if vehicle.vehicle_type == 'car' else SPEED_LIMIT_TRUCK_KMH

                if vehicle.speed_kmh > speed_limit:
                    speed_violations_count += 1

                if vehicle.speed_kmh > REALTIME_ALERT_SPEED_KMH:
                    alert_msg = f"REAL-TIME ALERT: Vehicle {vehicle.id} speed {vehicle.speed_kmh:.2f} km/h exceeds {REALTIME_ALERT_SPEED_KMH} km/h!"
                    logging.warning(alert_msg)
                    high_speed_alerts.append(alert_msg)

    cap.release()

    # Summarize results
    total_vehicles = len(vehicles)
    result = {
        "total_vehicles": total_vehicles,
        "speed_violations": speed_violations_count,
        "high_speed_alerts": high_speed_alerts,
    }

    logging.info(f"Video processing completed: {result}")

    return result
