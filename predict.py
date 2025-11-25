import sys
import json
import joblib
import pandas as pd
import numpy as np

MODEL_FILES = {
    "DecisionTreeClassifier": r"D:\predictingStudentsLate\DecisionTreeClassifier_model.pkl",
    "RandomForestClassifier": r"D:\predictingStudentsLate\best_model_final.pkl",
    "GradientBoostingClassifier": r"D:\predictingStudentsLate\GradientBoostingClassifier_model.pkl",
    "LogisticRegression": r"D:\predictingStudentsLate\LogisticRegression.pkl",
    "LinearSVC": r"D:\predictingStudentsLate\LinearSVC_model.pkl",
    "SVC": r"D:\predictingStudentsLate\SVC_model.pkl",
}

DEFAULT_MODEL = "RandomForestClassifier"


def _load_artifacts():
    models = {}

    for name, path in MODEL_FILES.items():
        artifact = joblib.load(path)

        if not isinstance(artifact, dict):
            raise ValueError(f"{name} artifact must be a dict with preprocessor & model")

        preprocessor = artifact.get("preprocessor")
        model = artifact.get("model")

        if preprocessor is None or model is None:
            raise ValueError(f"{name} artifact missing preprocessor or model")

        models[name] = {"preprocessor": preprocessor, "model": model}

    return models


try:
    ARTIFACTS = _load_artifacts()
except Exception as e:
    print(json.dumps({"error": "not load model", "detail": str(e)}))
    sys.exit(1)

def _calculate_probability(artifact, payload):
    df = pd.DataFrame([payload])
    X = artifact["preprocessor"].transform(df)
    model = artifact["model"]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))
    else:
        pred = model.predict(X)[0]
        prob = float(pred)

    return round(float(prob) * 100, 2)


def predict(data):
    payload = dict(data)
    model_name = payload.pop("model_name", DEFAULT_MODEL)

    if model_name not in ARTIFACTS:
        raise ValueError(f"invalid model_name '{model_name}'. Available: {', '.join(ARTIFACTS)}")

    comparisons = {
        name: _calculate_probability(artifact, payload)
        for name, artifact in ARTIFACTS.items()
    }

    return {
        "model_name": model_name,
        "probability": comparisons[model_name],
        "probabilities": comparisons,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "not data input"}))
        sys.exit(1)
        
    try:
        input_data = json.loads(sys.argv[1])
        result = predict(input_data)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))