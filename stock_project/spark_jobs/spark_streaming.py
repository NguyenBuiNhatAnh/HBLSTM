import sys
sys.path.insert(0, "/home/nhatanh/project/realtime_stock_HBLSTM")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
)
from collections import defaultdict

import time
import io
import json
import logging
import requests

import pandas as pd
import joblib
import torch
import numpy as np
from minio import Minio

from stock_project.models import LinearModel, DTModel, KNNModel, HBLSTMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("spark_streaming")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
KAFKA_TOPIC      = "stock-price"
KAFKA_SERVER     = "localhost:9092"
MINIO_ENDPOINT   = "localhost:9005"
ACCESS_KEY       = "admin"
SECRET_KEY       = "password123"
BACKEND_URL      = "http://localhost:8000/predictions/"
EMA_ALPHA        = 2 / (5 + 1)

# ──────────────────────────────────────────────
# KAFKA INPUT SCHEMA
# ──────────────────────────────────────────────
kafka_schema = StructType([
    StructField("symbol",   StringType(), True),
    StructField("datetime", StringType(), True),
    StructField("open",     DoubleType(), True),
    StructField("high",     DoubleType(), True),
    StructField("low",      DoubleType(), True),
    StructField("close",    DoubleType(), True),
    StructField("volume",   DoubleType(), True),
])

# ──────────────────────────────────────────────
# OUTPUT SCHEMA (yield từ process_partition)
# ──────────────────────────────────────────────
output_schema = StructType([
    StructField("symbol",       StringType(), True),
    StructField("datetime",     StringType(), True),
    StructField("actual_price", DoubleType(), True),
    StructField("pred_linear",  DoubleType(), True),
    StructField("pred_dt",      DoubleType(), True),
    StructField("pred_knn",     DoubleType(), True),
    StructField("pred_hblstm",  DoubleType(), True),
])

# ──────────────────────────────────────────────
# LOAD MODELS TỪ MINIO
# ──────────────────────────────────────────────
def load_models(symbol: str, client: Minio):
    def _pkl(path):
        return joblib.load(io.BytesIO(client.get_object("bucket-models", path).read()))

    linear = LinearModel()
    linear.model = _pkl(f"{symbol}/linear.pkl")

    dt = DTModel()
    dt.model = _pkl(f"{symbol}/decision_tree.pkl")

    knn = KNNModel()
    knn.model = _pkl(f"{symbol}/knn.pkl")

    scaler = _pkl(f"{symbol}/scaler.pkl")

    config = json.loads(
        client.get_object("bucket-models", f"{symbol}/config.json").read().decode()
    )

    hblstm = HBLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        seq_len=config["seq_len"],
    )
    hblstm.model.load_state_dict(
        torch.load(
            io.BytesIO(client.get_object("bucket-models", f"{symbol}/hblstm.pth").read()),
            map_location="cpu",
        )
    )
    hblstm.model.train()  # train mode để incremental_update hoạt động

    df_history = pd.read_parquet(
        io.BytesIO(client.get_object("bucket-data", f"{symbol}/historical.parquet").read())
    )

    models = {"linear": linear, "dt": dt, "knn": knn, "hblstm": hblstm}
    return models, scaler, config, df_history


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
                    timestamp: str):
    """
    POST một dự đoán của một model lên Django/FastAPI backend.
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
# PROCESS PARTITION (chạy trên executor)
# ──────────────────────────────────────────────
def process_partition(rows):
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    model_cache  = {}                  # symbol → (models, scaler, config)
    buffer_cache = defaultdict(list)   # symbol → list of scaled feature vectors
    ema_state    = {}                  # symbol → last EMA5
    row_cache    = defaultdict(list)   # symbol → raw rows để flush MinIO

    for row in rows:
        symbol = row.symbol

        # ── Load lazy ─────────────────────────────────────────────────────────
        if symbol not in model_cache:
            models, scaler, config, df_history = load_models(symbol, client)
            model_cache[symbol] = (models, scaler, config)
            ema_state[symbol] = float(df_history["EMA5"].iloc[-1])

        models, scaler, config = model_cache[symbol]
        seq_len    = config["seq_len"]
        n_features = config["input_size"]
        target_idx = 3  # index cột 'close'

        # ── EMA5 incremental ───────────────────────────────────────────────────
        ema_state[symbol] = EMA_ALPHA * row.close + (1 - EMA_ALPHA) * ema_state[symbol]
        ema5 = ema_state[symbol]

        # ── Timestamp numeric ──────────────────────────────────────────────────
        ts = float(pd.Timestamp(row.datetime).timestamp())

        # ── Feature vector (khớp features_cols lúc train offline) ─────────────
        # features_cols = ['open','high','low','close','volume','timestamp','EMA5']
        feat_vec = [
            float(row.open), float(row.high), float(row.low),
            float(row.close), float(row.volume), ts, ema5,
        ]

        # ── Scale ─────────────────────────────────────────────────────────────
        scaled_vec   = scaler.transform([feat_vec])[0]
        scaled_close = float(scaled_vec[target_idx])

        # ── Sliding window ─────────────────────────────────────────────────────
        buffer_cache[symbol].append(scaled_vec)
        if len(buffer_cache[symbol]) > seq_len:
            buffer_cache[symbol].pop(0)

        # ── Raw data accumulator ───────────────────────────────────────────────
        row_cache[symbol].append({
            "symbol": symbol, "datetime": row.datetime,
            "open": row.open, "high": row.high, "low": row.low,
            "close": row.close, "volume": row.volume, "EMA5": ema5,
        })

        # ── Predict khi đủ seq_len ─────────────────────────────────────────────
        if len(buffer_cache[symbol]) == seq_len:
            seq      = np.array(buffer_cache[symbol]).reshape(1, seq_len, n_features)
            seq_flat = seq.reshape(1, -1)

            p_linear = inverse_close(
                scaler, float(models["linear"].predict(seq_flat)[0]), n_features, target_idx
            )
            p_dt = inverse_close(
                scaler, float(models["dt"].predict(seq_flat)[0]), n_features, target_idx
            )
            p_knn = inverse_close(
                scaler, float(models["knn"].predict(seq_flat)[0]), n_features, target_idx
            )
            p_hblstm = inverse_close(
                scaler, float(models["hblstm"].predict(seq)[0]), n_features, target_idx
            )

            # Incremental update HBLSTM
            models["hblstm"].incremental_update(seq, np.array([[scaled_close]]))

            log.info(
                f"[{symbol}] {row.datetime} | actual={row.close:.4f} | "
                f"LR={p_linear:.4f} | DT={p_dt:.4f} | "
                f"KNN={p_knn:.4f} | HBLSTM={p_hblstm:.4f}"
            )

            yield (
                symbol,
                str(row.datetime),
                float(row.close),
                p_linear, p_dt, p_knn, p_hblstm,
            )

    # ── Flush raw data lên MinIO ───────────────────────────────────────────────
    for sym, records in row_cache.items():
        if not records:
            continue
        buf = io.BytesIO()
        pd.DataFrame(records).to_parquet(buf, index=False)
        buf.seek(0)
        client.put_object(
            "bucket-data",
            f"{sym}/stream_{int(time.time())}.parquet",
            buf, length=buf.getbuffer().nbytes,
        )


# ──────────────────────────────────────────────
# PROCESS BATCH (chạy trên driver)
# ──────────────────────────────────────────────
def process_batch(batch_df, epoch_id):
    if batch_df.rdd.isEmpty():
        log.info(f"[Batch {epoch_id}] Rỗng.")
        return

    rdd = batch_df.rdd.mapPartitions(process_partition)

    # Buffer chưa đủ seq_len → không có gì để predict
    if rdd.isEmpty():
        log.info(f"[Batch {epoch_id}] Warm-up – buffer chưa đủ seq_len.")
        return

    result_df = spark.createDataFrame(rdd, schema=output_schema)
    result_df.show(truncate=False)

    # ── POST 4 predictions / row lên backend ──────────────────────────────────
    rows = result_df.collect()

    MODEL_MAP = [
        ("LinearRegression", "pred_linear"),
        ("DecisionTree",     "pred_dt"),
        ("KNN",              "pred_knn"),
        ("HBLSTM",           "pred_hblstm"),
    ]

    for r in rows:
        for model_type, field in MODEL_MAP:
            post_prediction(
                symbol          = r.symbol,
                predicted_price = getattr(r, field),
                actual_price    = r.actual_price,
                model_type      = model_type,
                timestamp       = r.datetime,
            )

    # ── Lưu predictions batch lên MinIO ───────────────────────────────────────
    client = Minio(MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

    if not client.bucket_exists("bucket-predictions"):
        client.make_bucket("bucket-predictions")

    pdf       = result_df.toPandas()
    csv_bytes = pdf.to_csv(index=False).encode("utf-8")
    obj_name  = f"predictions_batch_{epoch_id}_{int(time.time())}.csv"

    client.put_object(
        "bucket-predictions", obj_name,
        io.BytesIO(csv_bytes), length=len(csv_bytes),
        content_type="text/csv",
    )
    log.info(
        f"[Batch {epoch_id}] {len(pdf)} rows → "
        f"MinIO (bucket-predictions/{obj_name}) & POST backend."
    )


# ──────────────────────────────────────────────
# SPARK SESSION
# ──────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("StockInference")
    .config("spark.sql.shuffle.partitions", "9")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ──────────────────────────────────────────────
# READ STREAM → PARSE JSON
# ──────────────────────────────────────────────
raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

parsed = (
    raw.selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), kafka_schema).alias("d"))
    .select("d.*")
    .filter(col("close").isNotNull())
)

# ──────────────────────────────────────────────
# WRITE STREAM
# ──────────────────────────────────────────────
query = (
    parsed.writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", "file:///tmp/spark-ckpt/stock-inference")
    .trigger(processingTime="5 seconds")
    .start()
)

log.info("Streaming started. Ctrl+C để dừng.")
query.awaitTermination()
