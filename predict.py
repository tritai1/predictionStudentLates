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


def predict_by_student_id(student_id, weekday, weather, model_name=None):
    """
    Dự đoán xác suất đi trễ cho một sinh viên với weekday và weather cụ thể
    
    Parameters:
    -----------
    student_id : str
        Mã sinh viên (phải có trong dataset)
    weekday : str
        Thứ ngày (Monday, Tuesday, ...)
    weather : str
        Thời tiết (sunny, rainy, cloudy, windy)
    model_name : str, optional
        Tên model để sử dụng. Nếu None sẽ dùng DEFAULT_MODEL
    
    Returns:
    --------
    dict: Kết quả dự đoán và thông tin sinh viên
    """
    import os
    
    # Load dataset
    dataset_path = r"D:\predictingStudentsLate\student_lateness_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df_train = pd.read_csv(dataset_path)
    
    # Kiểm tra student_id
    if student_id not in df_train['student_id'].values:
        available_students = df_train['student_id'].unique()[:10].tolist()
        raise ValueError(f"Student ID '{student_id}' not found in dataset. Sample IDs: {available_students}")
    
    # Lấy dữ liệu của sinh viên (lấy bản ghi đầu tiên làm mẫu)
    student_records = df_train[df_train['student_id'] == student_id]
    sample_record = student_records.iloc[0].copy()
    
    # Thay đổi weekday và weather
    sample_record['weekday'] = weekday
    sample_record['weather'] = weather
    
    # Tạo payload từ sample_record (bỏ student_id và is_late)
    payload = {
        col: sample_record[col] 
        for col in sample_record.index 
        if col not in ['student_id', 'is_late']
    }
    
    # Sử dụng model_name hoặc DEFAULT_MODEL
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in ARTIFACTS:
        raise ValueError(f"invalid model_name '{model_name}'. Available: {', '.join(ARTIFACTS)}")
    
    # Tính toán xác suất với tất cả models để so sánh
    comparisons = {
        name: _calculate_probability(artifact, payload)
        for name, artifact in ARTIFACTS.items()
    }
    
    # Thông tin sinh viên
    student_info = {
        'student_id': student_id,
        'total_records': int(len(student_records)),
        'actual_late_count': int((student_records['is_late'] == 'yes').sum()),
        'actual_on_time_count': int((student_records['is_late'] == 'no').sum()),
        'default_distance': float(sample_record['distance_km']),
        'default_transport': str(sample_record['transport_mode']),
        'default_speed': float(sample_record['avg_speed_kmh']),
    }
    
    # Chuyển đổi payload sang dict có thể serialize JSON
    input_data = {
        "weekday": weekday,
        "weather": weather,
    }
    
    for k, v in payload.items():
        if isinstance(v, (np.integer, np.int64)):
            input_data[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            input_data[k] = float(v)
        elif isinstance(v, (np.bool_, bool)):
            input_data[k] = bool(v)
        else:
            input_data[k] = str(v)
    
    return {
        "model_name": model_name,
        "probability": comparisons[model_name],
        "probabilities": comparisons,
        "student_info": student_info,
        "input_data": input_data
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "not data input"}))
        sys.exit(1)
        
    try:
        input_data = json.loads(sys.argv[1])
        
        # Kiểm tra nếu là dự đoán theo student_id
        if "student_id" in input_data and "weekday" in input_data and "weather" in input_data:
            student_id = input_data.pop("student_id")
            weekday = input_data.pop("weekday")
            weather = input_data.pop("weather")
            model_name = input_data.pop("model_name", None)
            result = predict_by_student_id(student_id, weekday, weather, model_name)
        else:
            # Dự đoán thông thường
            result = predict(input_data)
        
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))