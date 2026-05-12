from stock_project.models import LinearModel, DTModel, KNNModel, HBLSTMModel, LightGBMModel, CNNLSTMModel
import joblib
import torch
from minio import Minio
import io
import json
import numpy as np
import requests
import pandas as pd

BACKEND_URL = "http://backend:8000/predictions/"   # tên service trong compose


# ──────────────────────────────────────────────
# LOAD MODELS TỪ MINIO
# ──────────────────────────────────────────────
def load_models(symbol: str, client: Minio):

    # ==========================================
    # HELPER
    # ==========================================
    def _pkl(path):
        return joblib.load(
            io.BytesIO(
                client.get_object(
                    "bucket-models",
                    path
                ).read()
            )
        )

    # ==========================================
    # LOAD SKLEARN MODELS
    # ==========================================
    linear = LinearModel()
    linear.model = _pkl(f"{symbol}/linear.pkl")

    dt = DTModel()
    dt.model = _pkl(f"{symbol}/decision_tree.pkl")

    knn = KNNModel()
    knn.model = _pkl(f"{symbol}/knn.pkl")

    lightgbm = LightGBMModel()
    lightgbm.model = _pkl(f"{symbol}/lightgbm.pkl")

    # ==========================================
    # LOAD SCALER
    # ==========================================
    scaler = _pkl(f"{symbol}/scaler.pkl")

    # ==========================================
    # LOAD CONFIG
    # ==========================================
    config = json.loads(
        client.get_object(
            "bucket-models",
            f"{symbol}/config.json"
        ).read().decode()
    )

    # ==========================================
    # LOAD HBLSTM
    # ==========================================
    hblstm = HBLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hblstm_hidden_size"],
        seq_len=config["seq_len"],
    )

    hblstm.model.load_state_dict(
        torch.load(
            io.BytesIO(
                client.get_object(
                    "bucket-models",
                    f"{symbol}/hblstm.pth"
                ).read()
            ),
            map_location="cpu",
        )
    )

    # train mode cho incremental learning
    hblstm.model.train()

    # ==========================================
    # LOAD CNNLSTM
    # ==========================================
    cnnlstm = CNNLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["cnnlstm_hidden_size"],
        seq_len=config["seq_len"],
    )

    cnnlstm.model.load_state_dict(
        torch.load(
            io.BytesIO(
                client.get_object(
                    "bucket-models",
                    f"{symbol}/cnnlstm.pth"
                ).read()
            ),
            map_location="cpu",
        )
    )

    cnnlstm.model.eval()

    models = {
        "linear": linear,
        "dt": dt,
        "knn": knn,
        "lightgbm": lightgbm,
        "hblstm": hblstm,
        "cnnlstm": cnnlstm
    }

    return models, scaler, config

# ──────────────────────────────────────────────
# INVERSE TRANSFORM cột close
# ──────────────────────────────────────────────
def inverse_close(scaler, scaled_val: float, n_features: int, target_idx: int = 3) -> float:
    dummy = np.zeros((1, n_features))
    dummy[0, target_idx] = scaled_val
    return float(scaler.inverse_transform(dummy)[0, target_idx])

# ──────────────────────────────────────────────
# POST PREDICTION LÊN BACKEND
# ──────────────────────────────────────────────
def post_prediction(symbol: str, predicted_price: float,
                    actual_price: float, model_type: str,
                    timestamp: str, log):
    """
    POST một dự đoán của một model lên FastAPI backend.
    Không raise exception để không crash Spark.
    """
    payload = {
        "symbol":          symbol,
        "predicted_price": round(predicted_price, 6),
        "actual_price":    round(actual_price, 6),
        "model_type":      model_type,
        "timestamp":       timestamp,
    }
    try:
        resp = requests.post(BACKEND_URL, json=payload, timeout=5)
        if resp.status_code not in (200, 201):
            log.warning(
                f"POST [{model_type}] {symbol} → HTTP {resp.status_code}: {resp.text[:200]}"
            )
        else:
            log.info(f"POST [{model_type}] {symbol} @ {timestamp} → OK")
    except requests.exceptions.RequestException as e:
        log.error(f"POST failed [{model_type}] {symbol}: {e}")

# ──────────────────────────────────────────────
# LOAD EMA5
# ──────────────────────────────────────────────
def load_all_runtime_states(client):

    symbols = [
        'DJI',
        'SPX',
        'NDX',
        'DXY',
        'NI225',
        '000001',
        'NIFTY',
        'UKX',
        'DEU40',
    ]

    ema_states = {}

    for symbol in symbols:

        try:

            # ==========================================
            # LOAD ALL PARQUET FILES
            # ==========================================
            objects = client.list_objects(
                "bucket-data",
                prefix=f"{symbol}/",
                recursive=True
            )

            dfs = []

            for obj in objects:

                response = client.get_object(
                    "bucket-data",
                    obj.object_name
                )

                df = pd.read_parquet(
                    io.BytesIO(response.read())
                )

                dfs.append(df)

            if len(dfs) == 0:
                print(f"[WARN] No data for {symbol}")
                continue

            # ==========================================
            # CONCAT + SORT
            # ==========================================
            df_all = pd.concat(
                dfs,
                ignore_index=True
            )

            df_all["datetime"] = pd.to_datetime(
                df_all["datetime"]
            )

            df_all = df_all.sort_values(
                "datetime"
            )

            # ==========================================
            # GET LAST EMA5
            # ==========================================
            last_ema5 = float(
                df_all["EMA5"].iloc[-1]
            )

            ema_states[symbol] = last_ema5

            print(
                f"[EMA STATE] "
                f"{symbol} = {last_ema5:.4f}"
            )

        except Exception as e:

            print(
                f"[ERROR] {symbol}: {e}"
            )

    return ema_states