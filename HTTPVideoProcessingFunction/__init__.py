import azure.functions as func
import logging
import os
import tempfile
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from .processing import process_video_clip

# Configuration
INPUT_CONTAINER = "splitted-videos"
OUTPUT_CONTAINER = "processed-stats"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")

# Speed limits
SPEED_LIMITS = {
    "car": 90,
    "truck": 80,
    "emergency": 130
}

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger function called to process all split videos.")

    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        input_container = blob_service.get_container_client(INPUT_CONTAINER)

        processed_count = 0
        for blob in input_container.list_blobs():
            if not blob.name.endswith(".mp4"):
                continue

            logging.info(f"Processing blob: {blob.name}")

            # Download blob to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                stream = input_container.download_blob(blob.name)
                tmp.write(stream.readall())
                tmp_path = tmp.name

            try:
                results = process_video_clip(tmp_path)
                stats = generate_stats(blob.name, results)
                save_stats_to_blob(stats, blob.name)
                processed_count += 1
            except Exception as e:
                logging.error(f"Error processing {blob.name}: {str(e)}", exc_info=True)
            finally:
                os.remove(tmp_path)

        return func.HttpResponse(
            json.dumps({"message": f"Processed {processed_count} video(s)."}, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return func.HttpResponse(f"Server error: {str(e)}", status_code=500)


def generate_stats(video_name: str, results: dict) -> dict:
    stats = {
        "video_name": video_name,
        "processing_time": datetime.utcnow().isoformat(),
        "total_vehicles": len(results["vehicles"]),
        "speed_violations": {
            "cars": sum(1 for v in results["vehicles"] if v["type"] == "car" and v["speed"] > SPEED_LIMITS["car"]),
            "trucks": sum(1 for v in results["vehicles"] if v["type"] == "truck" and v["speed"] > SPEED_LIMITS["truck"])
        },
        "high_speed_alerts": [v for v in results["vehicles"] if v["speed"] > SPEED_LIMITS["emergency"]],
        "lane_stats": calculate_lane_stats(results),
        "time_stats": calculate_time_stats(results)
    }
    return stats


def calculate_lane_stats(results: dict) -> dict:
    lanes = {}
    for vehicle in results["vehicles"]:
        lane = vehicle["lane"]
        if lane not in lanes:
            lanes[lane] = {"count": 0, "speeds": []}
        lanes[lane]["count"] += 1
        lanes[lane]["speeds"].append(vehicle["speed"])
    for lane in lanes:
        lanes[lane]["avg_speed"] = sum(lanes[lane]["speeds"]) / len(lanes[lane]["speeds"])
    return lanes


def calculate_time_stats(results: dict) -> dict:
    time_stats = {}
    interval = 300  # 5 minutes
    for vehicle in results["vehicles"]:
        time_slot = int(vehicle["timestamp"] / interval) * interval
        if time_slot not in time_stats:
            time_stats[time_slot] = {"count": 0, "speeds": []}
        time_stats[time_slot]["count"] += 1
        time_stats[time_slot]["speeds"].append(vehicle["speed"])
    for slot in time_stats:
        time_stats[slot]["avg_speed"] = sum(time_stats[slot]["speeds"]) / len(time_stats[slot]["speeds"])
    return time_stats


def save_stats_to_blob(stats: dict, original_file_name: str):
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(OUTPUT_CONTAINER)
    if not container_client.exists():
        container_client.create_container()

    base_name = os.path.splitext(os.path.basename(original_file_name))[0]

    # JSON
    json_blob_name = f"{base_name}_stats.json"
    container_client.upload_blob(json_blob_name, json.dumps(stats, indent=2), overwrite=True)

    # TXT
    txt_blob_name = f"{base_name}_stats.txt"
    txt_content = generate_text_report(stats)
    container_client.upload_blob(txt_blob_name, txt_content, overwrite=True)


def generate_text_report(stats: dict) -> str:
    report = [
        f"Traffic Analysis Report",
        f"Video: {stats['video_name']}",
        f"Processed at: {stats['processing_time']}",
        "",
        f"Total vehicles detected: {stats['total_vehicles']}",
        f"Speed violations:",
        f"  - Cars > {SPEED_LIMITS['car']} km/h: {stats['speed_violations']['cars']}",
        f"  - Trucks > {SPEED_LIMITS['truck']} km/h: {stats['speed_violations']['trucks']}",
        "",
        "High speed alerts (>130 km/h):"
    ]

    for alert in stats["high_speed_alerts"]:
        report.append(
            f"  - {alert['type'].title()} at {alert['speed']:.1f} km/h (Lane: {alert['lane']}, Time: {alert['timestamp']:.1f}s)"
        )

    report.extend(["", "Lane Statistics:"])
    for lane, data in stats["lane_stats"].items():
        report.append(
            f"  - Lane {lane}: {data['count']} vehicles, Avg speed: {data['avg_speed']:.1f} km/h"
        )

    report.extend(["", "Time-based Statistics (5-minute intervals):"])
    for time_slot, data in stats["time_stats"].items():
        minutes = int(time_slot / 60)
        report.append(
            f"  - {minutes}-{minutes + 5} min: {data['count']} vehicles, Avg speed: {data['avg_speed']:.1f} km/h"
        )

    return "\n".join(report)
