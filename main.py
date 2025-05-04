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
with open("RF-balancedPalm.pkl", "rb") as f:
    model = pickle.load(f)

# دالة معالجة الصورة
def preprocess_image(image_data):
    try:
        # فتح الصورة
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # تغيير الحجم بما يتناسب مع المدخلات المدربة عليه
        image = image.resize((224, 224))  # تأكد إن ده هو الحجم اللي الموديل متدرب عليه
        
        # تحويل الصورة إلى مصفوفة numpy
        image_array = np.array(image) / 255.0  # تطبيع البيانات
        image_array = image_array.flatten().reshape(1, -1)  # تحويل الصورة إلى مصفوفة خطية
        return image_array
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# endpoint للتنبؤ
@app.post("/predict")
async def predict_anemia(file: UploadFile = File(...)):
    try:
        # قراءة محتويات الصورة
        contents = await file.read()
        
        # معالجة الصورة
        processed_image = preprocess_image(contents)
        
        # التنبؤ باستخدام الموديل
        prediction = model.predict(processed_image)
        
        # عرض النتيجة
        label = "Anemic" if prediction[0] == 1 else "Non-Anemic"
        return {"label": label}
    
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# مسار رئيسي للتأكد إن السيرفر شغال
@app.get("/")
def root():
    return {"message": "Anemia detection API is running!"}
