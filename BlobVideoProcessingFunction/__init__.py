import azure.functions as func
import logging
import os
import tempfile
from azure.storage.blob import BlobServiceClient
from .processing import process_video_clip

# Configuration
INPUT_CONTAINER = "splitted-videos"
OUTPUT_CONTAINER = "processed-stats"  # New container for statistics
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")

def main(blob: func.InputStream):
    # Validate input
    if not hasattr(blob, 'name') or not blob.name:
        logging.error("Invalid blob input - missing name property")
        return
    
    if not blob.length or blob.length == 0:
        logging.error(f"Empty blob detected: {blob.name}")
        return

    logging.info(f"Processing blob: {blob.name} (Size: {blob.length} bytes)")

    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(blob.read())
            tmp_path = tmp.name

        # Process video
        stats = process_video_clip(tmp_path)
        logging.info(f"Processing complete for {blob.name}")

        # Generate stats content
        stats_content = generate_stats_content(blob.name, stats)
        
        # Upload stats to the new container
        upload_stats_blob(blob.name, stats_content)
        
        # Cleanup
        os.remove(tmp_path)
        logging.info(f"Completed processing for {blob.name}")

    except Exception as e:
        logging.error(f"Failed to process {blob.name}: {str(e)}", exc_info=True)
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def generate_stats_content(blob_name: str, stats: dict) -> str:
    """Generate formatted stats content"""
    content = [
        f"Video: {blob_name}",
        f"Total vehicles detected: {stats.get('total_vehicles', 0)}",
        f"Speed violations: {stats.get('speed_violations', 0)}",
        "High speed alerts:"
    ]
    content.extend([f" - {alert}" for alert in stats.get('high_speed_alerts', [])])
    return "\n".join(content)

def upload_stats_blob(original_blob_name: str, content: str):
    """Upload stats to the dedicated output container"""
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("Azure Storage connection string not configured")
    
    try:
        # Generate stats blob name
        base_name = os.path.splitext(os.path.basename(original_blob_name))[0]
        stats_blob_name = f"{base_name}_stats.txt"
        
        # Create client and upload to the new container
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING)
        
        # Ensure the output container exists
        container_client = blob_service_client.get_container_client(OUTPUT_CONTAINER)
        if not container_client.exists():
            container_client.create_container()
            logging.info(f"Created container {OUTPUT_CONTAINER}")
        
        # Upload the stats file
        blob_client = container_client.get_blob_client(stats_blob_name)
        blob_client.upload_blob(content, overwrite=True)
        logging.info(f"Stats uploaded to {OUTPUT_CONTAINER}/{stats_blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to upload stats for {original_blob_name}: {str(e)}")
        raise