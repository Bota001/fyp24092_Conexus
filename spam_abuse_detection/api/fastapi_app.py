from fastapi import FastAPI
from pydantic import BaseModel
from scripts.inference import load_model

app = FastAPI()
classifier = load_model()

class Message(BaseModel):
    text: str

@app.post("/predict/")
def predict(message: Message):
    prediction = classifier([message.text])[0]
    return {"label": prediction["label"], "score": prediction["score"]}
