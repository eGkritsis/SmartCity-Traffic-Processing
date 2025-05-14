import azure.functions as func
import logging
import subprocess
import os
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger_video_split")
def http_trigger_video_split(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # 1. Define paths and settings
    input_video = "test.mp4"  # Video file must be in the same folder
    output_prefix = "split_"
    segment_time = "120"  # 2 minutes in seconds

    # Path to ffmpeg.exe located in the current working directory
    ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")

    # 2. Split video using FFmpeg
    try:
        subprocess.run([
            ffmpeg_path,
            "-i", input_video,
            "-c", "copy",
            "-map", "0",
            "-segment_time", segment_time,
            "-f", "segment",
            "-reset_timestamps", "1",
            f"{output_prefix}%03d.mp4"
        ], check=True)
        logging.info("Video successfully split into clips")
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed: {e}"
        logging.error(error_msg)
        return func.HttpResponse(error_msg, status_code=500)
    except FileNotFoundError as e:
        error_msg = f"FFmpeg not found. Expected at: {ffmpeg_path}"
        logging.error(error_msg)
        return func.HttpResponse(error_msg, status_code=500)

    # 3. Upload to Azure Blob Storage
    try:
        connection_string = os.environ["AzureWebJobsStorage"]
        container_name = "video-clips"

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Upload split files
        for file in os.listdir():
            if file.startswith(output_prefix) and file.endswith(".mp4"):
                with open(file, "rb") as data:
                    container_client.upload_blob(name=file, data=data)
                os.remove(file)
                logging.info(f"Uploaded {file} to blob storage")

        return func.HttpResponse("Video successfully split and uploaded to blob storage", status_code=200)

    except Exception as e:
        error_msg = f"Blob upload failed: {e}"
        logging.error(error_msg)
        return func.HttpResponse(error_msg, status_code=500)
