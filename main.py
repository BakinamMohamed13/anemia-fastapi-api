from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn
from PIL import Image
import io
import pickle

app = FastAPI()

# Load the model
with open("XGB-Tuned-balancedPalm.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "Anemia detection model is up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # Resize image to expected input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = image_array.reshape(1, -1)  # Flatten for XGBoost input

        prediction = model.predict(image_array)[0]
        label = "Anemia" if prediction == 1 else "Not Anemia"

        return {"prediction": label}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)