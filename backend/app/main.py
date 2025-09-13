from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model_utils import predict   # Import your model logic

app = FastAPI(title="Spam Detection API")

# Allow frontend requests (adjust port if your React dev server uses a different one)
origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Spam Detection API is running!"}

@app.post("/predict")
def predict_text(request: TextRequest):
    """Receive text and return spam prediction"""
    return predict(request.text)
