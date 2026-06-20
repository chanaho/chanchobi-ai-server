from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO

print("🚀 AI SERVER START")

app = FastAPI()

# =========================
# MODEL LOAD (딱 1번만)
# =========================
model = YOLO("best.pt")

print("✅ YOLO MODEL LOADED")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):
    try:
        contents = await file.read()

        return {
            "status": "success",
            "farm": farm,
            "result": {
                "disease": "test",
                "confidence": 50,
                "risk": "LOW"
            }
        }

    except Exception as e:
        return JSONResponse({
            "status": "failed",
            "error": str(e)
        })