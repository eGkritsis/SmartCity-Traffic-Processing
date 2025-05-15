import azure.functions as func
import logging
import subprocess
import os
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def get_ffmpeg_path():
    """Get the correct FFmpeg path for either local or Azure environment"""
    # Try system FFmpeg first
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "ffmpeg"
    except:
        # Check for FFmpeg in the local 'bin' directory
        local_ffmpeg = os.path.join(os.getcwd(), "bin", "ffmpeg")
        if os.path.exists(local_ffmpeg):
            return local_ffmpeg
        raise Exception("FFmpeg not found. Please ensure FFmpeg is installed or included in your deployment")

@app.route(route="http_trigger_video_split")
def http_trigger_video_split(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Video split function triggered.')

    # Configuration
    input_blob_name = "test.mp4"
    input_container = "raw-video"
    output_container = "splitted-videos"
    segment_time = "120"  # 2 minutes in seconds
    
    # Use forward slashes for Azure Functions (works on both Windows and Linux)
    temp_dir = "/tmp"
    input_video = f"{temp_dir}/{input_blob_name}"
    output_prefix = f"{temp_dir}/split_"

    try:
        # 1. Initialize Blob Service Client
        connection_string = os.environ["AzureWebJobsStorage"]
        blob_service = BlobServiceClient.from_connection_string(connection_string)

        # 2. Create /tmp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)

        # 3. Download video from blob storage
        logging.info(f"Downloading {input_blob_name} from {input_container}")
        blob_client = blob_service.get_blob_client(container=input_container, blob=input_blob_name)
        
        with open(input_video, "wb") as video_file:
            download_stream = blob_client.download_blob()
            video_file.write(download_stream.readall())
        logging.info("Download completed successfully")

        # 4. Get FFmpeg path and verify
        ffmpeg_cmd = get_ffmpeg_path()
        logging.info(f"Using FFmpeg at: {ffmpeg_cmd}")

        # 5. Split video using FFmpeg
        logging.info("Starting video splitting process")
        subprocess.run([
            ffmpeg_cmd,
            "-i", input_video,
            "-c", "copy",
            "-map", "0",
            "-segment_time", segment_time,
            "-f", "segment",
            "-reset_timestamps", "1",
            f"{output_prefix}%03d.mp4"
        ], check=True)
        logging.info("Video split completed")

        # 6. Upload split files to output container
        output_container_client = blob_service.get_container_client(output_container)
        
        for file in os.listdir(temp_dir):
            if file.startswith("split_") and file.endswith(".mp4"):
                file_path = f"{temp_dir}/{file}"
                with open(file_path, "rb") as data:
                    output_container_client.upload_blob(name=file, data=data)
                os.remove(file_path)
                logging.info(f"Uploaded {file}")

        # 7. Clean up
        if os.path.exists(input_video):
            os.remove(input_video)
        return func.HttpResponse("Video processed successfully", status_code=200)

    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)