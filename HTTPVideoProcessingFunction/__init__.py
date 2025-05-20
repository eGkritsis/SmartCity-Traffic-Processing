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
    """Generate summary stats: only cars and trucks, no duplicates per vehicle ID, keep highest speed"""
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj

    # Map from vehicle ID -> vehicle with highest speed
    vehicle_map = {}

    for vehicle in results["vehicles"]:
        vehicle_id = vehicle.get("id")
        vehicle_type = vehicle.get("type")
        
        if vehicle_type not in ["car", "truck"]:
            continue  # Skip other types

        # If not seen before OR this one is faster → keep it
        existing = vehicle_map.get(vehicle_id)
        if not existing or vehicle["speed"] > existing["speed"]:
            vehicle_map[vehicle_id] = vehicle

    # Convert to final deduplicated list
    unique_vehicles = [convert_numpy_types(v) for v in vehicle_map.values()]

    stats = {
        "video_name": video_name,
        "processing_time": datetime.utcnow().isoformat(),
        "total_vehicles": len(unique_vehicles),
        "vehicles": unique_vehicles,
        "video_metadata": convert_numpy_types(results.get("video_properties", {}))
    }

    logging.info(f"Generated deduplicated stats for {video_name}: {json.dumps(stats, cls=NumpyEncoder, indent=2)}")
    return stats


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
        
        logging.info(f"Saved stats for {original_blob_name} to {OUTPUT_CONTAINER}")

    except Exception as e:
        logging.error(f"Failed to save stats for {original_blob_name}: {str(e)}")
        raise