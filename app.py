from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

ner = pipeline(
    "ner",
    model="dannyLeo16/ner_model_bert_base",
    tokenizer="dannyLeo16/ner_model_bert_base"
)

@app.get("/")
def home():
    return {"message": "NLP Emergency API running"}

@app.post("/predict")
def predict(text: str):
    result = ner(text)
    return result
