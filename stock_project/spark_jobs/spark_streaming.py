import sys
sys.path.insert(0, "/home/nhatanh/project/realtime_stock_HBLSTM")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import *
from collections import defaultdict

import requests
import time
import pandas as pd
import json
import io
import joblib
import torch
import numpy as np
from minio import Minio
import os
from datetime import datetime

from stock_project.models import LinearModel, DTModel, KNNModel, HBLSTMModel

# =========================
# 1. CONFIG
# =========================
KAFKA_TOPIC = "stock-price"
KAFKA_SERVER = "localhost:9092"

MINIO_ENDPOINT = "localhost:9005"
ACCESS_KEY = "admin"
SECRET_KEY = "password123"

# =========================
# 2. SCHEMA KAFKA JSON
# =========================
schema = StructType([
    StructField("symbol", StringType()),
    StructField("datetime", StringType()),
    StructField("open", DoubleType()),
    StructField("high", DoubleType()),
    StructField("low", DoubleType()),
    StructField("close", DoubleType()),
    StructField("volume", DoubleType()),
])

# =========================
# 3. LOAD MODEL 1 LẦN / PARTITION
# =========================
def load_models(symbol, client):
    models = {}

    # ===== LINEAR =====
    obj = client.get_object("bucket-models", f"{symbol}/linear.pkl")
    linear = LinearModel()
    linear.model = joblib.load(io.BytesIO(obj.read()))
    models["linear"] = linear

    # ===== DT =====
    obj = client.get_object("bucket-models", f"{symbol}/decision_tree.pkl")
    dt = DTModel()
    dt.model = joblib.load(io.BytesIO(obj.read()))
    models["dt"] = dt  # Đã sửa key cho đồng nhất

    # ===== KNN =====
    obj = client.get_object("bucket-models", f"{symbol}/knn.pkl")
    knn = KNNModel()
    knn.model = joblib.load(io.BytesIO(obj.read()))
    models["knn"] = knn

    # ===== SCALER =====
    obj = client.get_object("bucket-models", f"{symbol}/scaler.pkl")
    scaler = joblib.load(io.BytesIO(obj.read()))

    # ===== CONFIG =====
    obj = client.get_object("bucket-models", f"{symbol}/config.json")
    config = json.loads(obj.read().decode())

    # ===== HBLSTM =====
    hblstm = HBLSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        seq_len=config["seq_len"]
    )

    obj = client.get_object("bucket-models", f"{symbol}/hblstm.pth")
    buffer = io.BytesIO(obj.read())
    hblstm.model.load_state_dict(torch.load(buffer, map_location="cpu"))
    hblstm.model.train()  # 🔥 QUAN TRỌNG: Để train() vì ta đang incremental update
    models["hblstm"] = hblstm
    
    # ===== HISTORY DATA (Tùy chọn: dùng để mồi buffer ban đầu) =====
    obj = client.get_object("bucket-data", f"{symbol}/historical.parquet")
    buffer = io.BytesIO(obj.read())
    df_history = pd.read_parquet(buffer)

    return models, scaler, config, df_history

# =========================
# 4. PROCESS PARTITION
# =========================
def process_partition(rows):

    client = Minio(
        MINIO_ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )

    # =========================
    # STATE
    # =========================
    model_cache = {}
    buffer_cache = defaultdict(list)   # scaled seq buffer
    ema_state = {}
    row_cache = defaultdict(list)      # raw storage buffer

    alpha = 2 / (5 + 1)

    # =========================
    # PROCESS STREAM BATCH
    # =========================
    for row in rows:

        symbol = row.symbol

        # =========================
        # LOAD MODEL LAZY PER SYMBOL
        # =========================
        if symbol not in model_cache:
            models, scaler, config, df_history = load_models(symbol, client)
            model_cache[symbol] = (models, scaler, config)

            # init EMA từ history
            ema_state[symbol] = df_history["EMA5"].iloc[-1]

        models, scaler, config = model_cache[symbol]

        # =========================
        # TIME FEATURE
        # =========================
        dt_obj = pd.to_datetime(row.datetime)
        timestamp_val = int(dt_obj.timestamp())

        # =========================
        # EMA UPDATE (STATEFUL)
        # =========================
        prev_ema = ema_state[symbol]
        new_ema = alpha * row.close + (1 - alpha) * prev_ema
        ema_state[symbol] = new_ema

        # =========================
        # RAW RECORD (FOR MINIO)
        # =========================
        raw_record = {
            "symbol": symbol,
            "datetime": row.datetime,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "timestamp": timestamp_val,
            "ema5": new_ema
        }

        row_cache[symbol].append(raw_record)

        # =========================
        # FEATURE VECTOR (FOR MODEL)
        # =========================
        features = [
            row.open,
            row.high,
            row.low,
            row.close,
            row.volume,
            timestamp_val,
            new_ema
        ]

        # scale
        features_scaled = scaler.transform([features])[0]

        # buffer sliding window
        buffer_cache[symbol].append(features_scaled)

        if len(buffer_cache[symbol]) > config["seq_len"]:
            buffer_cache[symbol].pop(0)

        # =========================
        # PREDICT + UPDATE MODEL
        # =========================
        if len(buffer_cache[symbol]) == config["seq_len"]:

            seq = np.array(buffer_cache[symbol]).reshape(1, config["seq_len"], -1)
            seq_flat = seq.reshape(1, -1)

            models_dict, _, _ = model_cache[symbol]

            p_linear = float(models_dict["linear"].predict(seq_flat)[0])
            p_dt = float(models_dict["dt"].predict(seq_flat)[0])
            p_knn = float(models_dict["knn"].predict(seq_flat)[0])
            p_hblstm = float(models_dict["hblstm"].predict(seq)[0])

            # incremental update HBLSTM
            x_new = seq
            y_new = np.array([[row.close]])
            models_dict["hblstm"].incremental_update(x_new, y_new)

            # yield result for Spark
            yield (
                symbol,
                row.datetime,
                p_linear,
                p_dt,
                p_knn,
                p_hblstm
            )

    # =========================
    # FLUSH RAW DATA TO MINIO
    # =========================
    for symbol, records in row_cache.items():

        if not records:
            continue

        df = pd.DataFrame(records)

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        client.put_object(
            "bucket-data",
            f"{symbol}/stream_{int(time.time())}.parquet",
            buffer,
            length=buffer.getbuffer().nbytes
        )

# =========================
# 5. MAIN SPARK & BATCH PROCESS
# =========================
spark = SparkSession.builder \
    .appName("StockInference") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_SERVER) \
    .option("subscribe", KAFKA_TOPIC) \
    .load()

parsed = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

def process_batch(batch_df, epoch_id):
    # Áp dụng mapPartitions
    rdd = batch_df.rdd.mapPartitions(process_partition)
    
    if rdd.isEmpty():
        return

    # Định nghĩa schema rõ ràng
    schema_output = ["symbol", "datetime", "pred_linear", "pred_dt", "pred_knn", "pred_hblstm"]
    result_df = spark.createDataFrame(rdd, schema=schema_output)
    
    # Hiển thị log terminal
    print(f"--- Batch {epoch_id} ---")
    result_df.show()

    # 🔥 LƯU PREDICTION VÀO MINIO 🔥
    client = Minio(
        MINIO_ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=False
    )
    
    # Chuyển đổi qua Pandas để lưu dạng file CSV lên MinIO cho từng batch
    pdf = result_df.toPandas()
    csv_bytes = pdf.to_csv(index=False).encode('utf-8')
    
    client.put_object(
        "bucket-predictions", # Đảm bảo bạn đã tạo bucket này trên MinIO
        f"predictions_batch_{epoch_id}.csv",
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="application/csv"
    )

query = parsed.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()