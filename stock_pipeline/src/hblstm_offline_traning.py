import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import json

# Import từ các file utils của bạn
from module.data_processor import preprocess_data, calculate_ema5
from module.dataset import TimeSeriesDataset, create_sliding_windows
from module.model import HBLSTM
from module.train_evaluate_hblstm import train_model, evaluate_model
from module.minio_helper import MinIOHelper # Import class MinIOHelper mới tạo

# ==========================
# 1. CÀI ĐẶT THÔNG SỐ (HYPERPARAMETERS)
# ==========================
seq_len = 3         # Nhìn lại 3 cây nến
batch_size = 16
hidden_size = 32
num_epochs = 50
learning_rate = 0.001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Đang chạy trên thiết bị: {device}")

# ==========================
# 2. XỬ LÝ DỮ LIỆU & CHUẨN HÓA (ĐÃ FIX DATA LEAKAGE)
# ==========================
print("Đang tải và xử lý dữ liệu...")
df_raw = pd.read_csv('000001.csv')

# Áp dụng các hàm tiền xử lý
df_clean = preprocess_data(df_raw)
df_ema = calculate_ema5(df_clean)

# Chọn các cột để làm Features (Đầu vào)
features_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA5']
data_values = df_ema[features_cols].values 
input_size = len(features_cols) # Sẽ là 6
    
# Xác định cột mục tiêu (Dự đoán giá 'close' -> index là 3)
target_col_idx = 3

# --- BẮT ĐẦU SỬA DATA LEAKAGE ---
# Xác định vị trí cắt mảng 2D thành Train/Test (80%)
split_idx_raw = int(len(data_values) * 0.8)

# Khởi tạo Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# BƯỚC QUAN TRỌNG: Chỉ dùng hàm fit() để dò tìm Min/Max trên tập Train
scaler.fit(data_values[:split_idx_raw])

# Sau đó mới áp dụng transform() để chuẩn hóa cho toàn bộ mảng dữ liệu
scaled_data = scaler.transform(data_values)
# --- KẾT THÚC SỬA DATA LEAKAGE ---

# ==========================
# 3. TẠO DATASET & DATALOADER
# ==========================
X, y = create_sliding_windows(scaled_data, target_col_idx, seq_len)

# Train/Test Split (Chia theo thứ tự thời gian, 80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Mẹo: Bật shuffle=True ở tập Train sẽ giúp mô hình hội tụ tốt hơn!
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==========================
# 4. KHỞI TẠO MODEL, LOSS, OPTIMIZER
# ==========================
model = HBLSTM(input_size=input_size, hidden_size=hidden_size).to(device)
criterion = nn.MSELoss() # Hàm mất mát Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ==========================
# 5. VÒNG LẶP TRAIN & TEST
# ==========================
print("==========================")
print("Bắt đầu quá trình huấn luyện...")
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
        
    if (epoch + 1) % 10 == 0:
        test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

# ==========================
# 6. LƯU CÁC FILE CẦN THIẾT TẠI LOCAL
# ==========================
print("==========================")
print("ĐANG LƯU FILE TẠI LOCAL...")

# 6.1 Lưu trọng số Model
torch.save(model.state_dict(), 'hblstm_model.pth')

# 6.2 Lưu Scaler
joblib.dump(scaler, 'scaler.pkl')

# 6.3 Tạo và lưu file Config
config = {
    "seq_len": seq_len,
    "input_size": input_size,
    "hidden_size": hidden_size,
    "features_cols": features_cols,
    "target_col_idx": target_col_idx
}
with open('model_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4)

# 6.4 Lưu Data lịch sử dạng Parquet
df_ema.to_parquet('historical_data.parquet', index=False)
print("Đã lưu xong 4 file: hblstm_model.pth, scaler.pkl, model_config.json, historical_data.parquet")

# ==========================
# 7. UPLOAD TỰ ĐỘNG LÊN MINIO
# ==========================
print("==========================")
print("ĐANG UPLOAD LÊN MINIO...")
# Đảm bảo port 9005 đúng với cấu hình docker-compose của bạn
uploader = MinIOHelper(endpoint="localhost:9005", access_key="admin", secret_key="password123")

# Đẩy 3 file liên quan đến Model vào bucket-models
uploader.upload_file("bucket-models", "hblstm_model.pth")
uploader.upload_file("bucket-models", "scaler.pkl")
uploader.upload_file("bucket-models", "model_config.json")

# Đẩy Dữ liệu lịch sử vào bucket-data
uploader.upload_file("bucket-data", "historical_data.parquet")

print("==========================")
print("🎉 QUÁ TRÌNH OFFLINE PIPELINE ĐÃ HOÀN TẤT TRỌN VẸN!")