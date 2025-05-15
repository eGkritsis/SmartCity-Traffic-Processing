import azure.functions as func
import logging
import os
import tempfile
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from .processing import process_video_clip
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Configuration
INPUT_CONTAINER = "splitted-videos"
OUTPUT_CONTAINER = "processed-stats"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")

# Speed limits (km/h)
SPEED_LIMITS = {
    "car": 90,
    "truck": 80,
    "emergency": 130
}

def main(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger function to process all split videos"""
    logging.info("HTTP trigger function processing all videos in container")
    
    try:
        # Initialize Blob Service Client
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(INPUT_CONTAINER)
        
        if not container_client.exists():
            return func.HttpResponse(
                f"Container {INPUT_CONTAINER} not found",
                status_code=404
            )

        # Process each video in the container
        processed_files = []
        failed_files = []
        
        for blob in container_client.list_blobs():
            if not blob.name.endswith(".mp4"):
                continue

            logging.info(f"Processing video: {blob.name}")
            
            try:
                # Download blob to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    blob_client = container_client.get_blob_client(blob.name)
                    download_stream = blob_client.download_blob()
                    tmp_file.write(download_stream.readall())
                    tmp_path = tmp_file.name

                # Process video with enhanced tracking
                results = process_video_clip(tmp_path)
                
                if not results or len(results.get("vehicles", [])) == 0:
                    logging.warning(f"No vehicles detected in {blob.name}")
                    continue

                # Generate and save stats (using custom encoder)
                stats = generate_stats(blob.name, results)
                save_stats_to_blob(stats, blob.name)
                processed_files.append(blob.name)
                
                # Clean up
                os.remove(tmp_path)

            except Exception as e:
                logging.error(f"Failed to process {blob.name}: {str(e)}", exc_info=True)
                failed_files.append(blob.name)
                continue

        # Prepare response
        response = {
            "status": "completed",
            "processed_files": processed_files,
            "failed_files": failed_files,
            "processed_count": len(processed_files),
            "failed_count": len(failed_files),
            "completion_time": datetime.utcnow().isoformat()
        }

        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return func.HttpResponse(
            f"Server error: {str(e)}",
            status_code=500
        )

def generate_stats(video_name: str, results: dict) -> dict:
    """Generate comprehensive statistics with numpy type handling"""
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    # Filter out invalid speeds and empty records
    valid_vehicles = [
        convert_numpy_types(v) for v in results["vehicles"] 
        if v.get("speed", 0) > 5 and v.get("speed", 0) < 300  # 5-300 km/h valid range
    ]
    
    stats = {
        "video_name": video_name,
        "processing_time": datetime.utcnow().isoformat(),
        "total_vehicles": len(valid_vehicles),
        "speed_violations": {
            "cars": sum(1 for v in valid_vehicles 
                      if v.get("type") == "car" and v.get("speed", 0) > SPEED_LIMITS["car"]),
            "trucks": sum(1 for v in valid_vehicles 
                       if v.get("type") == "truck" and v.get("speed", 0) > SPEED_LIMITS["truck"])
        },
        "high_speed_alerts": [
            v for v in valid_vehicles 
            if v.get("speed", 0) > SPEED_LIMITS["emergency"]
        ],
        "lane_stats": calculate_lane_stats(valid_vehicles),
        "time_stats": calculate_time_stats(valid_vehicles),
        "video_metadata": convert_numpy_types(results.get("video_properties", {}))
    }
    
    # Log stats without numpy types
    logging.info(f"Generated stats for {video_name}: {json.dumps(stats, cls=NumpyEncoder, indent=2)}")
    return stats

# [Rest of your functions (calculate_lane_stats, calculate_time_stats, save_stats_to_blob, generate_text_report) remain the same]
# Just make sure to use json.dumps(stats, cls=NumpyEncoder) when saving JSON

def calculate_lane_stats(vehicles: list) -> dict:
    """Calculate per-lane statistics with validation"""
    lanes = {}
    for vehicle in vehicles:
        lane = vehicle.get("lane", "unknown")
        speed = vehicle.get("speed", 0)
        
        if lane not in lanes:
            lanes[lane] = {"count": 0, "speeds": []}
        
        lanes[lane]["count"] += 1
        lanes[lane]["speeds"].append(speed)
    
    # Calculate averages with validation
    for lane in lanes:
        if len(lanes[lane]["speeds"]) > 0:
            lanes[lane]["avg_speed"] = sum(lanes[lane]["speeds"]) / len(lanes[lane]["speeds"])
        else:
            lanes[lane]["avg_speed"] = 0
    
    return lanes

def calculate_time_stats(vehicles: list) -> dict:
    """Calculate time-based statistics (5-minute intervals)"""
    time_stats = {}
    interval = 300  # 5 minutes in seconds
    
    for vehicle in vehicles:
        timestamp = vehicle.get("timestamp", 0)
        speed = vehicle.get("speed", 0)
        time_slot = int(timestamp / interval) * interval
        
        if time_slot not in time_stats:
            time_stats[time_slot] = {"count": 0, "speeds": []}
        
        time_stats[time_slot]["count"] += 1
        time_stats[time_slot]["speeds"].append(speed)
    
    # Calculate averages with validation
    for slot in time_stats:
        if len(time_stats[slot]["speeds"]) > 0:
            time_stats[slot]["avg_speed"] = sum(time_stats[slot]["speeds"]) / len(time_stats[slot]["speeds"])
        else:
            time_stats[slot]["avg_speed"] = 0
    
    return time_stats

def save_stats_to_blob(stats: dict, original_blob_name: str):
    """Save statistics to output container with error handling"""
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
        
        if not container_client.exists():
            container_client.create_container()
            logging.info(f"Created container {OUTPUT_CONTAINER}")

        # Generate blob names
        base_name = os.path.splitext(os.path.basename(original_blob_name))[0]
        
        # Save as JSON
        json_blob_name = f"{base_name}_stats.json"
        json_client = container_client.get_blob_client(json_blob_name)
        json_client.upload_blob(json.dumps(stats, indent=2), overwrite=True)
        
        # Save as TXT
        txt_blob_name = f"{base_name}_stats.txt"
        txt_content = generate_text_report(stats)
        txt_client = container_client.get_blob_client(txt_blob_name)
        txt_client.upload_blob(txt_content, overwrite=True)
        
        logging.info(f"Saved stats for {original_blob_name} to {OUTPUT_CONTAINER}")

    except Exception as e:
        logging.error(f"Failed to save stats for {original_blob_name}: {str(e)}")
        raise

def generate_text_report(stats: dict) -> str:
    """Generate human-readable report"""
    report = [
        "TRAFFIC ANALYSIS REPORT",
        "=======================",
        f"Video: {stats['video_name']}",
        f"Processing time: {stats['processing_time']}",
        "",
        f"Total valid vehicles detected: {stats['total_vehicles']}",
        "",
        "SPEED VIOLATIONS:",
        f"- Cars exceeding {SPEED_LIMITS['car']} km/h: {stats['speed_violations']['cars']}",
        f"- Trucks exceeding {SPEED_LIMITS['truck']} km/h: {stats['speed_violations']['trucks']}",
        "",
        "HIGH SPEED ALERTS (>130 km/h):"
    ]
    
    for alert in stats["high_speed_alerts"]:
        report.append(
            f"- {alert.get('type', 'vehicle').title()} at {alert.get('speed', 0):.1f} km/h "
            f"(Lane: {alert.get('lane', 'unknown')}, Time: {alert.get('timestamp', 0):.1f}s)"
        )
    
    report.extend(["", "LANE STATISTICS:"])
    for lane, data in stats["lane_stats"].items():
        report.append(
            f"- Lane {lane}: {data['count']} vehicles, "
            f"Avg speed: {data['avg_speed']:.1f} km/h"
        )
    
    report.extend(["", "TIME-BASED STATISTICS (5-minute intervals):"])
    for time_slot, data in stats["time_stats"].items():
        minutes = int(time_slot / 60)
        report.append(
            f"- {minutes}-{minutes+5} min: {data['count']} vehicles, "
            f"Avg speed: {data['avg_speed']:.1f} km/h"
        )
    
    return "\n".join(report)