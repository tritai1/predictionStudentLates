import sys
import json
import joblib
import pandas as pd

# Load file đã lưu (chứa cả preprocessor + model)
try:
    artifacts = joblib.load(r'D:\predictingStudentsLate\best_model_final.pkl')
    preprocessor = artifacts['preprocessor']
    model = artifacts['model']
except Exception as e:
    print(json.dumps({"error": "not load model", "detail": str(e)}))
    sys.exit(1)

def predict(data):
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    return round(float(prob) * 100, 2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "not data input"}))
        sys.exit(1)
        
    try:
        input_data = json.loads(sys.argv[1])
        result = predict(input_data)
        print(json.dumps({"probability": result}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))