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

symbols = [
    'DJI', 'SPX', 'NDX',
    'DXY', 'NI225', '000001',
    'NIFTY', 'UKX', 'DEU40'
]

def get_minio_client():
    return Minio(
        "stock-minio:9000",
        access_key="admin",
        secret_key="password123",
        secure=False
    )

def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print(f"  {name} -> MSE: {mse:.6f} | RMSE: {rmse:.6f}")

def load_symbol_data(client: Minio, symbol: str) -> pd.DataFrame:
    """Load toàn bộ parquet files của 1 symbol từ bucket-data rồi concat lại."""
    objects = client.list_objects("bucket-data", prefix=f"{symbol}/", recursive=True)

    dfs = []
    for obj in objects:
        response = client.get_object("bucket-data", obj.object_name)
        df = pd.read_parquet(io.BytesIO(response.read()))
        dfs.append(df)

    if not dfs:
        raise ValueError(f"Không tìm thấy dữ liệu nào cho symbol: {symbol}")

    df_history = pd.concat(dfs, ignore_index=True)
    return df_history

def upload_sklearn_model(client: Minio, model_obj, symbol: str, name: str):
    buffer = io.BytesIO()
    joblib.dump(model_obj, buffer)
    buffer.seek(0)
    client.put_object(
        "bucket-models",
        f"{symbol}/{name}.pkl",
        buffer,
        length=buffer.getbuffer().nbytes
    )

def upload_pytorch_model(client: Minio, state_dict, symbol: str):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    client.put_object(
        "bucket-models",
        f"{symbol}/hblstm.pth",
        buffer,
        length=buffer.getbuffer().nbytes
    )

def upload_scaler(client: Minio, scaler, symbol: str):
    buffer = io.BytesIO()
    joblib.dump(scaler, buffer)
    buffer.seek(0)
    client.put_object(
        "bucket-models",
        f"{symbol}/scaler.pkl",
        buffer,
        length=buffer.getbuffer().nbytes
    )

def upload_config(client: Minio, config: dict, symbol: str):
    config_bytes = json.dumps(config).encode("utf-8")
    client.put_object(
        "bucket-models",
        f"{symbol}/config.json",
        io.BytesIO(config_bytes),
        length=len(config_bytes)
    )

def ensure_buckets(client: Minio):
    for bucket in ["bucket-models", "bucket-data"]:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

def train_all_model():
    """
    DAG callable: load toàn bộ dữ liệu từ MinIO, retrain 4 model cho mỗi
    trong 9 symbol, rồi upload model + scaler + config trở lại MinIO.
    """
    client = get_minio_client()
    ensure_buckets(client)

    features_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'EMA5']
    target_col_idx = 3   # 'close'
    seq_len = 3
    input_size = len(features_cols)  # 7

    for symbol in symbols:
        print(f"\n{'='*40}")
        print(f"[{symbol}] Bắt đầu retrain...")

        # ------------------------------------------------------------------
        # 1. LOAD & TIỀN XỬ LÝ DỮ LIỆU
        # ------------------------------------------------------------------
        try:
            df_history = load_symbol_data(client, symbol)
        except ValueError as e:
            print(f"  SKIP - {e}")
            continue

        # Dữ liệu đã qua preprocess + EMA5 khi upload lần đầu,
        # nhưng ta vẫn gọi lại để đảm bảo dữ liệu mới nhất sạch đúng chuẩn.
        df_clean = preprocess_data(df_history)
        df_ema5  = calculate_ema5(df_clean)

        data_values = df_ema5[features_cols].values

        # ------------------------------------------------------------------
        # 2. CHUẨN HÓA
        # ------------------------------------------------------------------
        split_idx_raw = int(len(data_values) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_values[:split_idx_raw])          # fit chỉ trên train
        scaled_data = scaler.transform(data_values)

        # ------------------------------------------------------------------
        # 3. SLIDING WINDOW & TRAIN/TEST SPLIT
        # ------------------------------------------------------------------
        X, y = create_sliding_windows(
            scaled_data,
            target_col='close',
            feature_cols=features_cols,
            seq_len=seq_len
        )

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat  = X_test.reshape(X_test.shape[0], -1)

        # ------------------------------------------------------------------
        # 4. TRAIN
        # ------------------------------------------------------------------
        linear_regression = LinearModel()
        decision_tree     = DTModel()
        knn               = KNNModel()
        hblstm            = HBLSTMModel()

        print(f"  Training Linear Regression...")
        linear_regression.train(X_train_flat, y_train)

        print(f"  Training Decision Tree...")
        decision_tree.train(X_train_flat, y_train)

        print(f"  Training KNN...")
        knn.train(X_train_flat, y_train)

        print(f"  Training HBLSTM...")
        hblstm.train(X_train, y_train)

        # ------------------------------------------------------------------
        # 5. EVALUATE
        # ------------------------------------------------------------------
        print(f"  --- Evaluation [{symbol}] ---")
        evaluate(y_test, linear_regression.predict(X_test_flat), "Linear Regression")
        evaluate(y_test, decision_tree.predict(X_test_flat),     "Decision Tree")
        evaluate(y_test, knn.predict(X_test_flat),               "KNN")
        evaluate(y_test, hblstm.predict(X_test),                 "HBLSTM")

        # ------------------------------------------------------------------
        # 6. UPLOAD LÊN MINIO
        # ------------------------------------------------------------------
        print(f"  Uploading models → MinIO [{symbol}]...")

        upload_sklearn_model(client, linear_regression.model, symbol, "linear")
        upload_sklearn_model(client, decision_tree.model,     symbol, "decision_tree")
        upload_sklearn_model(client, knn.model,               symbol, "knn")
        upload_pytorch_model(client, hblstm.model.state_dict(), symbol)
        upload_scaler(client, scaler, symbol)
        upload_config(client, {
            "seq_len":       seq_len,
            "input_size":    input_size,
            "hidden_size":   hblstm.hidden_size,
            "features_cols": features_cols,
            "target_col":    "close"
        }, symbol)

        print(f"  Done [{symbol}] ✓")

    print(f"\n{'='*40}")
    print("Retrain toàn bộ hoàn tất!")