import joblib

_model = joblib.load("models/model.joblib")

def classify_text(text: str):
    label = _model.predict([text])[0]
    probs = None
    if hasattr(_model, "predict_proba"):
        classes = list(getattr(_model, "classes_", []))
        vals = _model.predict_proba([text])[0]
        probs = dict(zip(map(str, classes), map(float, vals)))
    return {"label": str(label), "probs": probs}
