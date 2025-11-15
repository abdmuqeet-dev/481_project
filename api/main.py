from fastapi import FastAPI
from pydantic import BaseModel
from predict import classify_text  # make sure predict.py exists

app = FastAPI(title="Stock News Sentiment API")

class Item(BaseModel):
    text: str

@app.get("/")
def root():
    return {"ok": True, "service": "stock-news-sentiment"}

@app.post("/classify")
def classify(item: Item):
    return classify_text(item.text)
