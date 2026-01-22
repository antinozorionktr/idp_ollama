from fastapi import FastAPI, UploadFile
import torch
from detectron2.engine import DefaultPredictor

app = FastAPI()

@app.get("/health")
def health():
    return {
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0)
    }

@app.post("/infer")
async def infer(file: UploadFile):
    return {"status": "ok"}  # placeholder
