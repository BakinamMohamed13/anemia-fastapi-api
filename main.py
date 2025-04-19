from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import io
from PIL import Image
import pickle

app = FastAPI()

# فعل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تقديم الملفات الساكنة (زي index.html)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# تحميل النموذج
with open("XGB-Tuned-balancedPalm.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))
    img_flatten = img_resized.flatten().reshape(1, -1)

    prediction = model.predict(img_flatten)
    result = "Anemia Detected" if prediction[0] == 1 else "Normal"
    return JSONResponse(content={"prediction": result})
