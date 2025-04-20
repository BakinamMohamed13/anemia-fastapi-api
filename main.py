from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import pickle

app = FastAPI()

# CORS to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # غيّري دي لو عندك URL محدد
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pickle model
with open("XGB-Tuned-balancedpalm.pk1", "rb") as f:  # <-- غيّري الاسم لو مختلف
    model = pickle.load(f)

# Image preprocessing
def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))  # ممكن تعدّلي المقاس حسب اللي الموديل محتاجه
    image_array = np.array(image) / 255.0  # normalize to 0-1
    flat = image_array.flatten().reshape(1, -1)  # Flatten to 1D
    return flat

# API Endpoint
@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image)
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"
        return {"label": label}
    except Exception as e:
        return {"error": str(e)}
        from fastapi.responses import HTMLResponse

@app.get("/")
async def serve_html():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

