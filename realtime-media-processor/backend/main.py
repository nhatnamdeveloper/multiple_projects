from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Realtime Media Processor Backend",
    description="Backend for real-time image, video, and livestream processing with ML inference",
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

# --- ML Model Loading ---
# Load the pre-trained MobileNetV2 model
try:
    logger.info("Loading MobileNetV2 model...")
    model = MobileNetV2(weights="imagenet")
    model.trainable = False  # Freeze the model weights
    logger.info("MobileNetV2 model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading MobileNetV2 model: {e}", exc_info=True)
    model = None # Set model to None if loading fails

@app.on_event("startup")
async def startup_event():
    if model is None:
        logger.error("ML model not loaded. /predict/image endpoint will not function.")

# --- Helper functions for image processing and prediction ---
def preprocess_image_for_model(image: Image.Image):
    """Resizes and preprocesses the image for MobileNetV2 input."""
    image = image.resize((224, 224))  # MobileNetV2 expects 224x224 input
    image_array = np.asarray(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return preprocess_input(image_array) # Preprocess for MobileNetV2

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Realtime Media Processor Backend!"}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, performs ML inference using MobileNetV2, and returns predictions.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not loaded. Please check backend logs."
        )

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB") # Ensure RGB
        
        logger.info(f"Received image: {file.filename}, dimensions: {image.size}")

        # Preprocess image for the model
        processed_image = preprocess_image_for_model(image)
        
        # Perform inference
        predictions = model.predict(processed_image)
        
        # Decode predictions (ImageNet classes)
        decoded_predictions = decode_predictions(predictions.numpy(), top=5)[0] # Top 5 predictions

        # Format prediction data
        formatted_predictions = []
        for _, label, confidence in decoded_predictions:
            formatted_predictions.append({
                "label": label,
                "confidence": float(confidence),
                "image_dimensions": {"height": image.height, "width": image.width}
            })
        
        logger.info(f"Processed image: {file.filename}, predictions: {formatted_predictions}")

        return JSONResponse(content={
            "filename": file.filename,
            "predictions": formatted_predictions,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image: {e}"
        )

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