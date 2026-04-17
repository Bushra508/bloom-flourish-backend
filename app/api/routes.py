from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from app.utils.preprocessing import preprocess
from app.services.inference import predict

router = APIRouter()

@router.get("/")
def root():
    return {"message": "API is running"}

@router.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        tensor = preprocess(image)
        results = predict(tensor)

        return {
            "success": True,
            "top_prediction": results[0],
            "all_predictions": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }