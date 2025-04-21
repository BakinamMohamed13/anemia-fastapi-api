import xgboost as xgb
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import pickle

app = FastAPI()

# السماح للفرونت إند بالتواصل مع الباك إند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # تقدر تحدد الدومين لو عايز
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الموديل باستخدام pickle
with open("XGB-Tuned-balancedPalm.pkl", "rb") as f:
    model = pickle.load(f)

# دالة معالجة الصورة
def preprocess_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))  # تأكد إن ده هو الحجم اللي الموديل متدرب عليه
        image_array = np.array(image) / 255.0  # تطبيع البيانات
        flat = image_array.flatten().reshape(1, -1)
        return flat
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# endpoint للتنبؤ
@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        prediction = model.predict(processed_image)
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"
        return {"label": label}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# مسار رئيسي للتأكد إن السيرفر شغال
@app.get("/")
def root():
    return {"message": "Anemia detection API is running!"}
