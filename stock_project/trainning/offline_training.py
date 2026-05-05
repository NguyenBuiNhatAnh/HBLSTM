import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stock_project.models import LinearModel, DTModel, KNNModel, HBLSTMModel
from stock_project.data.data_preprocessing import calculate_ema5, preprocess_data
from stock_project.data.dataset import create_sliding_windows
from sklearn.metrics import mean_squared_error
import numpy as np
import io
import json
import joblib
from minio import Minio
import torch

def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"{name} -> MSE: {mse:.6f} | RMSE: {rmse:.6f}")


# Cấu hình đường dẫn
data_paths = [
    {'symbol': 'DJI',     'path': 'stock_project/data/stock_data/DJI.csv'},
    {'symbol': 'SPX',     'path': 'stock_project/data/stock_data/SPX.csv'},
    {'symbol': 'NDX',     'path': 'stock_project/data/stock_data/NDX.csv'},
    {'symbol': 'DXY',     'path': 'stock_project/data/stock_data/DXY.csv'},
    {'symbol': 'NI225',   'path': 'stock_project/data/stock_data/NI225.csv'},
    {'symbol': '000001',  'path': 'stock_project/data/stock_data/000001.csv'},
    {'symbol': 'NIFTY',   'path': 'stock_project/data/stock_data/NIFTY.csv'},
    {'symbol': 'UKX',     'path': 'stock_project/data/stock_data/UKX.csv'},
    {'symbol': 'DEU40',   'path': 'stock_project/data/stock_data/DEU40.csv'},
]



for index in data_paths:
    
    
    print(f"Đang tải dữ liệu {index['symbol']}")
    df_raw = pd.read_csv(index['path'])
    
    # ==========================
    # 1. XỬ LÝ DỮ LIỆU & CHUẨN HÓA
    # ==========================
    
    # Tiền xử lý dữ liệu
    df_clean = preprocess_data(df_raw)
    # Tính ema5
    df_ema5 = calculate_ema5(df_clean)
    
    # Chọn các cột để làm Features (Đầu vào)
    features_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'EMA5']
    data_values = df_ema5[features_cols].values 
    input_size = len(features_cols) # Sẽ là 7
    
    # Xác định cột mục tiêu (Dự đoán giá 'close' -> index là 3)
    target_col_idx = 3
    
    # Xác định vị trí cắt mảng 2D thành Train/Test (80%)
    split_idx_raw = int(len(data_values) * 0.8)
    # Khởi tạo Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # BƯỚC QUAN TRỌNG: Chỉ dùng hàm fit() để dò tìm Min/Max trên tập Train
    scaler.fit(data_values[:split_idx_raw])
    # Sau đó mới áp dụng transform() để chuẩn hóa cho toàn bộ mảng dữ liệu
    scaled_data = scaler.transform(data_values)
    
    # ==========================
    # 2. TẠO DATASET & DATALOADER
    # ==========================
    
    X, y = create_sliding_windows(
        scaled_data,
        target_col='close',
        feature_cols=features_cols,
        seq_len=3
    )
    
    # Train/Test Split (Chia theo thứ tự thời gian, 80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    
    # =================
    # 4. KHỞI TẠO MODEL
    # =================
    
    linear_regression = LinearModel()
    decision_tree = DTModel()
    knn = KNNModel()
    hblstm = HBLSTMModel()
    
    # Train 4 model
    print(f"Đang tải train LR trên {index['symbol']}")
    linear_regression.train(X_train_flat, y_train)
    print(f"Đang tải train DT trên {index['symbol']}")
    decision_tree.train(X_train_flat, y_train)
    print(f"Đang tải train KNN trên {index['symbol']}")
    knn.train(X_train_flat, y_train)
    print(f"Đang tải train HBLSTM trên {index['symbol']}")
    hblstm.train(X_train, y_train)
    
    # Test 4 model
    # ===== SKLEARN =====
    lr_preds = linear_regression.predict(X_test_flat)
    dt_preds = decision_tree.predict(X_test_flat)
    knn_preds = knn.predict(X_test_flat)

    # ===== HBLSTM =====
    hblstm_preds = hblstm.predict(X_test)
    
    evaluate(y_test, lr_preds, "Linear Regression")
    evaluate(y_test, dt_preds, "Decision Tree")
    evaluate(y_test, knn_preds, "KNN")
    evaluate(y_test, hblstm_preds, "HBLSTM")
    
    # ==========================
    # 5. UPLOAD TỰ ĐỘNG LÊN MINIO
    # ==========================
    
    client = Minio(
        "localhost:9005",
        access_key="admin",
        secret_key="password123",
        secure=False
    )

    # tạo bucket nếu chưa có
    for bucket in ["bucket-models", "bucket-data"]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
    
    symbol = index['symbol']

    print(f"Đang upload lên MinIO: {symbol}")
    
    # ==========================
    # 5.1. Upload Models
    # ==========================
    models = {
        "linear": linear_regression.model,
        "decision_tree": decision_tree.model,
        "knn": knn.model
    }
    
    # sklearn models (.pkl)
    for name, model_obj in models.items():
        buffer = io.BytesIO()
        joblib.dump(model_obj, buffer)
        buffer.seek(0)

        client.put_object(
            "bucket-models",
            f"{symbol}/{name}.pkl",
            buffer,
            length=buffer.getbuffer().nbytes
        )
    
    # HBLSTM (.pth)
    buffer = io.BytesIO()
    torch.save(hblstm.model.state_dict(), buffer)
    buffer.seek(0)

    client.put_object(
        "bucket-models",
        f"{symbol}/hblstm.pth",
        buffer,
        length=buffer.getbuffer().nbytes
    )
    
    # ==========================
    # 5.2. Upload Scaler
    # ==========================
    buffer = io.BytesIO()
    joblib.dump(scaler, buffer)
    buffer.seek(0)

    client.put_object(
        "bucket-models",
        f"{symbol}/scaler.pkl",
        buffer,
        length=buffer.getbuffer().nbytes
    )
    
    # ==========================
    # 5.3. UPLOAD CONFIG
    # ==========================
    config = {
        "seq_len": 3,
        "input_size": input_size,
        "hidden_size": hblstm.hidden_size,
        "features_cols": features_cols,
        "target_col": "close"
    }

    config_bytes = json.dumps(config).encode("utf-8")

    client.put_object(
        "bucket-models",
        f"{symbol}/config.json",
        io.BytesIO(config_bytes),
        length=len(config_bytes)
    )
    
    # ==========================
    # 5.4. UPLOAD DATA (PARQUET)
    # ==========================

    buffer = io.BytesIO()
    df_ema5.to_parquet(buffer, index=False)
    buffer.seek(0)

    client.put_object(
        "bucket-data",
        f"{symbol}/historical.parquet",
        buffer,
        length=buffer.getbuffer().nbytes
    )

    print(f"Upload xong: {symbol}")
    print("--------------------------")
    
        
    
    
        