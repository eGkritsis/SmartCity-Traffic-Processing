import os
import logging
import json
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
import azure.functions as func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get environment variables
        endpoint = os.environ['CosmosDBEndpoint']
        key = os.environ['CosmosDBKey']
        blob_connection_string = os.environ['AzureWebJobsStorage']

        # Initialize Cosmos DB client
        client = CosmosClient(endpoint, key)
        database_name = "traffic-analysis"  
        container_name = "video_stats"

        # Get database and container references
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        # Initialize Blob Storage client
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client("processed-stats")

        processed_count = 0
        error_count = 0
        blobs = container_client.list_blobs()

        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue

            try:
                logger.info(f"Processing blob: {blob.name}")

                # Download and parse JSON
                blob_client = container_client.get_blob_client(blob)
                json_data = json.loads(blob_client.download_blob().readall())

                # Prepare Cosmos DB document
                document = {
                    "id": json_data["video_name"].replace(".mp4", ""),
                    "video_name": json_data["video_name"],
                    "partitionKey": json_data["video_name"],
                    "processing_time": json_data["processing_time"],
                    "total_vehicles": json_data["total_vehicles"],
                    "vehicles": json_data["vehicles"],  # List of vehicle dicts
                    "video_metadata": json_data["video_metadata"]
                }

                # Upsert to Cosmos DB
                container.upsert_item(document)
                processed_count += 1
                logger.info(f"Successfully processed {blob.name}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {blob.name}: {str(e)}")
                error_count += 1
            except cosmos_exceptions.CosmosHttpResponseError as e:
                logger.error(f"Cosmos DB error with {blob.name}: {str(e)}")
                error_count += 1
            except AzureError as e:
                logger.error(f"Azure Storage error with {blob.name}: {str(e)}")
                error_count += 1
            except Exception as e:
                logger.error(f"Unexpected error with {blob.name}: {str(e)}")
                error_count += 1

        return func.HttpResponse(
            json.dumps({
                "status": "completed",
                "processed": processed_count,
                "errors": error_count,
                "message": f"Successfully processed {processed_count} files with {error_count} errors"
            }),
            status_code=200,
            mimetype="application/json"
        )

    except KeyError as e:
        logger.error(f"Missing environment variable: {str(e)}")
        return func.HttpResponse(
            f"Configuration error: Missing environment variable {str(e)}",
            status_code=500
        )
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return func.HttpResponse(
            f"Fatal error: {str(e)}",
            status_code=500
        )