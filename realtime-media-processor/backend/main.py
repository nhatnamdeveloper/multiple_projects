from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from PIL import Image
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Realtime Media Processor Backend",
    description="Backend for real-time image, video, and livestream processing",
    version="0.1.0",
)

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Realtime Media Processor Backend!"}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, processes it (placeholder), and returns a dummy prediction.
    """
    try:
        # Create a temporary directory to save the uploaded file
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Received image: {file.filename}, saved to {file_path}")

        # --- Placeholder for ML Inference ---
        # In a real application, you would load your ML model here (e.g., TensorFlow, PyTorch)
        # and perform inference on the image.
        
        # Open and process the image with PIL (or OpenCV)
        image = Image.open(file_path)
        # Example: Convert to numpy array and get shape
        image_np = np.array(image)
        height, width, channels = image_np.shape
        
        # Dummy prediction data
        dummy_prediction = {
            "class": "detected_object",
            "confidence": 0.95,
            "bounding_box": [10, 20, 100, 150], # [x_min, y_min, x_max, y_max]
            "image_dimensions": {"height": height, "width": width}
        }
        logger.info(f"Processed image: {file.filename}, dummy prediction: {dummy_prediction}")
        # --- End Placeholder ---

        # Clean up the temporary file
        os.remove(file_path)
        shutil.rmtree(upload_dir) # Remove the directory after processing

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": dummy_prediction,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")

# Placeholder for video stream endpoint (e.g., using WebSockets)
# @app.websocket("/ws/video")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             # Process video frame (e.g., decode, run inference)
#             # Send results back
#             await websocket.send_json({"status": "processing_frame", "results": "..."})
#     except WebSocketDisconnect:
#         logger.info("Client disconnected from video websocket")
#     except Exception as e:
#         logger.error(f"Error in video websocket: {e}")
#         raise HTTPException(status_code=500, detail=f"Video streaming error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
