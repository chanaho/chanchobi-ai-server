import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

print("MAIN LOADED")
print("FILE:", __file__)
print("WORKDIR:", os.getcwd())

app = FastAPI()


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "server running"
    }


@app.get("/health")
def health():
    return {
        "status": "running"
    }


def run_ai(selected_crop):
    return {
        "ai_crop": selected_crop,
        "disease": "정상",
        "disease_confidence": 99.0,
        "risk": "LOW",
        "recommend": "Render 안정 테스트 단계",
        "crop_match": True
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    selected_crop: str = Form("unknown"),
    farm: str = Form("unknown")
):
    try:
        await file.read()

        result = run_ai(selected_crop)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )