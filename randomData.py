# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta

# # Set random seed for new data
# np.random.seed(789)  # Đảm bảo dữ liệu khác với các file trước

# # Parameters
# n_rows = 15000
# departments = ['DCCN', 'DCDK', 'DCVT', 'DCIT', 'DCME']
# weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# weather_options = ['sunny', 'rainy', 'cloudy', 'windy']
# weather_probs = [0.5, 0.2, 0.2, 0.1]

# # Generate data
# data = {
#     'student_id': [f'N22{np.random.choice(departments)}{np.random.randint(0, 1000):03d}' for _ in range(n_rows)],
#     'weekday': np.random.choice(weekdays, size=n_rows),
#     'class_start_time': ['07:00'] * n_rows,
#     'distance_km': np.random.uniform(0.5, 10.0, n_rows).round(1),
#     'weather': np.random.choice(weather_options, p=weather_probs, size=n_rows),
# }

# df = pd.DataFrame(data)

# # Assign transport mode based on distance
# def assign_transport(distance):
#     if distance < 2:
#         return 'walk'
#     elif distance <= 5:
#         return 'bike'
#     else:
#         return 'bus'

# df['transport_mode'] = df['distance_km'].apply(assign_transport)

# # Assign average speed based on transport mode
# def assign_speed(transport):
#     if transport == 'walk':
#         return np.random.uniform(4, 6)
#     elif transport == 'bike':
#         return np.random.uniform(15, 25)
#     else:  # bus
#         return np.random.uniform(20, 40)

# df['avg_speed_kmh'] = df['transport_mode'].apply(assign_speed).round(1)

# # Other columns
# df['habitual_punctuality'] = np.random.uniform(0.5, 1.0, n_rows).round(2)
# df['sleep_hours'] = np.random.uniform(5.0, 9.0, n_rows).round(2)
# df['alarm_used'] = np.random.choice(['yes', 'no'], p=[0.8, 0.2], size=n_rows)

# # Calculate expected_time, check_in_time, late_minutes
# def calculate_times(row):
#     class_start = datetime.strptime('07:00', '%H:%M')
#     travel_time_hours = row['distance_km'] / row['avg_speed_kmh']
#     travel_time_minutes = travel_time_hours * 60
#     expected_time = class_start - timedelta(minutes=travel_time_minutes)
#     delay_minutes = np.random.uniform(0, 15)
#     check_in_time = expected_time + timedelta(minutes=delay_minutes)
#     late_minutes = max(0, (check_in_time - class_start).total_seconds() / 60)
#     return pd.Series({
#         'expected_time': expected_time.strftime('%H:%M'),
#         'check_in_time': check_in_time.strftime('%H:%M'),
#         'late_minutes': round(late_minutes)
#     })

# # Apply time calculations
# time_cols = df.apply(calculate_times, axis=1)
# df = pd.concat([df, time_cols], axis=1)

# # Save to CSV
# df.to_csv('student_punctuality_new_15000.csv', index=False)
# print("Generated 15,000 rows with 13 columns and saved to 'student_punctuality_new_15000.csv'")
# print("Columns:", df.columns.tolist())


import pandas as pd
import numpy as np
import random
from scipy.stats import norm

# 1. Khởi tạo Kích thước
N_ROWS = 50000
N_STUDENTS = 500 # 100 quan sát cho mỗi sinh viên

# 2. Tạo các cột phân loại
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weather_conditions = ['sunny', 'cloudy', 'rainy', 'windy']
transport_modes = ['walk', 'bike', 'bus', 'car', 'train']
traffic_states = ['light', 'moderate', 'heavy']
importance_levels = ['low', 'medium', 'high']
alarm_options = ['yes', 'no']

data = {}
data['student_id'] = [f'N22DCQCN{str(i).zfill(3)}' for i in range(1, N_STUDENTS + 1)] * (N_ROWS // N_STUDENTS)

# Tạo các đặc trưng ngẫu nhiên
data['weekday'] = np.random.choice(weekdays, N_ROWS, p=[0.25, 0.2, 0.2, 0.2, 0.15])
data['weather'] = np.random.choice(weather_conditions, N_ROWS, p=[0.5, 0.3, 0.15, 0.05])
data['transport_mode'] = np.random.choice(transport_modes, N_ROWS, p=[0.1, 0.25, 0.3, 0.2, 0.15])
data['alarm_used'] = np.random.choice(alarm_options, N_ROWS, p=[0.8, 0.2])
data['hour_start_time'] = np.random.choice([7, 8, 9, 10, 13, 14, 15, 16], N_ROWS, p=[0.15, 0.2, 0.15, 0.1, 0.15, 0.1, 0.1, 0.05])
data['class_importance'] = np.random.choice(importance_levels, N_ROWS, p=[0.2, 0.5, 0.3])
data['unexpected_event'] = np.random.choice(['yes', 'no'], N_ROWS, p=[0.08, 0.92]) # Sự kiện bất ngờ hiếm khi xảy ra

# 3. Tạo các cột số thực có tính toán
df = pd.DataFrame(data)

# Khoảng cách nhà (phân phối lệch phải, ít người ở quá xa)
df['distance_km'] = np.round(np.clip(norm.rvs(loc=10, scale=8, size=N_ROWS), 1, 30), 1)

# Thói quen đúng giờ (phân phối chuẩn quanh 0.7)
df['habitual_punctuality'] = np.round(np.clip(norm.rvs(loc=0.7, scale=0.2, size=N_ROWS), 0.1, 1.0), 2)

# Giờ ngủ (phân phối chuẩn quanh 7.0)
df['sleep_hours'] = np.round(np.clip(norm.rvs(loc=7.0, scale=1.5, size=N_ROWS), 4.0, 10.0), 1)

# Thời gian chuẩn bị (phân phối hơi lệch, thường là 20-40 phút)
df['preparation_time'] = np.round(np.clip(norm.rvs(loc=30, scale=15, size=N_ROWS), 10, 60), 0)

# Tốc độ trung bình (phụ thuộc vào phương tiện)
mode_speeds = {'walk': 5.0, 'bike': 18.0, 'bus': 25.0, 'car': 35.0, 'train': 50.0}
df['avg_speed_kmh'] = df['transport_mode'].apply(lambda x: np.round(np.clip(norm.rvs(loc=mode_speeds[x], scale=mode_speeds[x]*0.15, size=1)[0], 5, 60), 1))

# Tình trạng giao thông (phụ thuộc vào giờ học và ngày)
def generate_traffic(row):
    # Giờ cao điểm
    if row['hour_start_time'] in [7, 8, 9, 16] and row['weekday'] in ['Monday', 'Friday']:
        return np.random.choice(['moderate', 'heavy'], p=[0.5, 0.5])
    # Giờ thấp điểm
    if row['hour_start_time'] in [10, 15]:
        return np.random.choice(['light', 'moderate'], p=[0.8, 0.2])
    # Các giờ khác
    return np.random.choice(['light', 'moderate', 'heavy'], p=[0.4, 0.5, 0.1])

df['traffic_condition'] = df.apply(generate_traffic, axis=1)

# 4. Tính toán Nhãn Mục tiêu (is_late)
df['TTC_hours'] = df['distance_km'] / df['avg_speed_kmh'] # Thời gian di chuyển cơ bản

# Định nghĩa các hệ số trễ/thưởng
penalties = {
    'traffic': {'light': -0.05, 'moderate': 0.1, 'heavy': 0.3},
    'weather': {'sunny': -0.05, 'cloudy': 0.0, 'windy': 0.1, 'rainy': 0.2},
    'importance': {'low': 0.1, 'medium': 0.0, 'high': -0.1},
    'event': {'yes': 0.5, 'no': 0.0}
}

# Áp dụng các hệ số
df['LF_Traffic'] = df['traffic_condition'].map(penalties['traffic'])
df['LF_Weather'] = df['weather'].map(penalties['weather'])
df['LF_Importance'] = df['class_importance'].map(penalties['importance'])
df['LF_Event'] = df['unexpected_event'].map(penalties['event'])

# Bonus/Penalty từ thói quen và giấc ngủ
df['LF_Punctuality'] = (1 - df['habitual_punctuality']) * 0.3 # Trễ nhiều hơn nếu thói quen đúng giờ kém
df['LF_Sleep'] = (8.0 - df['sleep_hours']) * 0.05             # Trễ nhiều hơn nếu ngủ ít hơn 8 tiếng

# Tổng Hệ số Trễ (Lateness Factor)
df['Lateness_Factor'] = (
    df['LF_Traffic'] + df['LF_Weather'] + 
    df['LF_Importance'] + df['LF_Event'] + 
    df['LF_Punctuality'] + df['LF_Sleep']
)

# Tính Thời gian Di chuyển Thực tế (Actual Travel Time - ATT)
# ATT = TTC_hours * (1 + Lateness_Factor)
# Đặt giới hạn cho Lateness_Factor để ATT không quá vô lý
df['ATT_hours'] = df['TTC_hours'] * (1 + np.clip(df['Lateness_Factor'], -0.5, 2.0))

# Tổng Thời gian Cần thiết (Total Time Required)
df['TTR_hours'] = df['ATT_hours'] + (df['preparation_time'] / 60)

# Giả định sinh viên rời nhà vào 7:00AM, nhưng chúng ta chỉ quan tâm đến thời gian có sẵn
# Thời gian có sẵn để di chuyển và chuẩn bị = (Giờ học - 0)
# Đây là một mô hình đơn giản: Giả sử sinh viên đi muộn nếu TTR_hours > 1.2 giờ (72 phút) và TTR này cao hơn thời gian còn lại trước giờ học.

# Mặc dù không có "thời gian rời nhà" rõ ràng, ta sử dụng TTR_hours để dự đoán trễ
# Ngưỡng trễ đơn giản (ví dụ: TTR trên 72 phút tương đương đi trễ 30% thời gian)
THRESHOLD_TTR = 1.2 # Ngưỡng (giờ) cho TTR
df['is_late'] = np.where(df['TTR_hours'] > THRESHOLD_TTR, 'yes', 'no')

# Tinh chỉnh nhỏ để đảm bảo sinh viên ngủ ít + giao thông/thời tiết tệ bị trễ nhiều hơn
high_risk = (df['sleep_hours'] < 6) & (df['traffic_condition'] == 'heavy')
df.loc[high_risk, 'is_late'] = np.random.choice(['yes', 'no'], size=high_risk.sum(), p=[0.7, 0.3])


# 5. Hoàn thiện
final_columns = [
    'student_id', 'weekday', 'distance_km', 'weather', 'transport_mode', 
    'avg_speed_kmh', 'habitual_punctuality', 'sleep_hours', 'alarm_used', 
    'hour_start_time', 'traffic_condition', 'preparation_time', 
    'unexpected_event', 'class_importance', 'is_late'
]

df_final = df[final_columns].copy()


df_final.to_csv(r'D:\predictingStudentsLate\student_lateness_dataset.csv', index=False)