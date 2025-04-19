from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import pickle

app = FastAPI()

# إضافة CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ده بيخلي السيرفر يقبل الطلبات من أي مكان
    allow_credentials=True,
    allow_methods=["*"],  # بيخلي السيرفر يقبل كل أنواع الـ HTTP methods
    allow_headers=["*"],  # بيخلي السيرفر يقبل كل أنواع الـ headers
)

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
