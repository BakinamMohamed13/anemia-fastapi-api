from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import xgboost as xgb
from PIL import Image
import numpy as np
import io
import uvicorn

app = FastAPI()

# CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change if you have a specific URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the XGBoost model (ensure you're using the correct version of XGBoost)
model = xgb.Booster()
model.load_model("XGB-Tuned-balancedPalm.pkl")  # Update the file name if needed

# Image preprocessing function
def preprocess_image(image_data):
    try:
        # Open the image and convert it to RGB format
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Resize the image to the required input size for the model
        image = image.resize((224, 224))  # Adjust size based on your model
        # Convert image to numpy array and normalize it (0-1 range)
        image_array = np.array(image) / 255.0
        # Flatten the image to a 1D array as the model expects
        flat = image_array.flatten().reshape(1, -1)
        return flat
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# API Endpoint for prediction
@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        # Predict using the XGBoost model
        prediction = model.predict(xgb.DMatrix(processed_image))
        # Convert prediction to a human-readable label
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"
        return {"label": label}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# Start the server when the script is run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # Ensure port 10000 is used
