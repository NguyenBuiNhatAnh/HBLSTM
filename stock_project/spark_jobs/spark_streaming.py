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
import logging

import pandas as pd
import numpy as np
from minio import Minio
# from support_func import load_models, inverse_close, post_prediction, load_all_runtime_states
from stock_project.support_func import *



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("spark_streaming")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# KAFKA_TOPIC      = "stock-price"
# KAFKA_SERVER     = "localhost:9092"
# MINIO_ENDPOINT   = "localhost:9005"
# ACCESS_KEY       = "admin"
# SECRET_KEY       = "password123"
# BACKEND_URL      = "http://localhost:8000/predictions/"
# EMA_ALPHA        = 2 / (5 + 1)

KAFKA_TOPIC      = "stock-price"
KAFKA_SERVER     = "kafka:29092"
MINIO_ENDPOINT   = "stock-minio:9000"   # dùng tên container + port nội bộ
ACCESS_KEY       = "admin"
SECRET_KEY       = "password123"
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
# PROCESS PARTITION (chạy trên executor)
# ──────────────────────────────────────────────
def process_partition(rows):
    client = Minio(
        MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False
    )

    model_cache  = {}                  # symbol → (models, scaler, config)
    buffer_cache = defaultdict(list)   # symbol → list of scaled feature vectors
    ema_state = load_all_runtime_states(client)                  # symbol → last EMA5
    row_cache    = defaultdict(list)   # symbol → raw rows để flush MinIO

    for row in rows:
        symbol = row.symbol

        # ── Load lazy ─────────────────────────────────────────────────────────
        if symbol not in model_cache:
            models, scaler, config = load_models(symbol, client)
            model_cache[symbol] = (models, scaler, config)

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
            p_lightgbm = inverse_close(
                scaler, float(models["lightgbm"].predict(seq_flat)[0]), n_features, target_idx
            )
            p_hblstm = inverse_close(
                scaler, float(models["hblstm"].predict(seq)[0]), n_features, target_idx
            )
            p_cnnlstm = inverse_close(
                scaler, float(models["cnnlstm"].predict(seq)[0]), n_features, target_idx
            )

            # Incremental update HBLSTM
            models["hblstm"].incremental_update(seq, np.array([[scaled_close]]))

            # ==========================================
            # LOGGING
            # ==========================================
            log.info(
                f"[{symbol}] {row.datetime} | "
                f"actual={row.close:.4f} | "
                f"LR={p_linear:.4f} | "
                f"DT={p_dt:.4f} | "
                f"KNN={p_knn:.4f} | "
                f"LGBM={p_lightgbm:.4f} | "
                f"HBLSTM={p_hblstm:.4f} | "
                f"CNNLSTM={p_cnnlstm:.4f}"
            )

            # ==========================================
            # OUTPUT
            # ==========================================
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
# PROCESS BATCH (driver)
# ──────────────────────────────────────────────
def process_batch(batch_df, epoch_id):

    # ==========================================
    # EMPTY CHECK
    # ==========================================
    if batch_df.rdd.isEmpty():

        log.info(
            f"[Batch {epoch_id}] Empty batch."
        )

        return

    # ==========================================
    # PARTITION PROCESSING
    # ==========================================
    rdd = batch_df.rdd.mapPartitions(
        process_partition
    )

    # ==========================================
    # WARMUP CHECK
    # ==========================================
    if rdd.isEmpty():

        log.info(
            f"[Batch {epoch_id}] "
            f"Warm-up phase "
            f"(buffer < seq_len)."
        )

        return

    # ==========================================
    # CREATE RESULT DATAFRAME
    # ==========================================
    result_df = spark.createDataFrame(
        rdd,
        schema=output_schema
    )

    result_df.show(truncate=False)

    # ==========================================
    # COLLECT RESULTS
    # ==========================================
    rows = result_df.collect()

    # ==========================================
    # MODEL FIELD MAPPING
    # ==========================================
    MODEL_MAP = [
        ("LinearRegression", "pred_linear"),
        ("DecisionTree",     "pred_dt"),
        ("KNN",              "pred_knn"),
        ("LightGBM",         "pred_lightgbm"),
        ("HBLSTM",           "pred_hblstm"),
        ("CNNLSTM",          "pred_cnnlstm"),
    ]

    # ==========================================
    # POST TO BACKEND
    # ==========================================
    for r in rows:

        for model_type, field in MODEL_MAP:

            try:

                post_prediction(
                    symbol          = r.symbol,
                    predicted_price = float(getattr(r, field)),
                    actual_price    = float(r.actual_price),
                    model_type      = model_type,
                    timestamp       = r.datetime,
                    log             = log
                )

            except Exception as e:

                log.error(
                    f"[POST ERROR] "
                    f"{r.symbol} "
                    f"{model_type}: {e}"
                )

    # ==========================================
    # SAVE PREDICTIONS TO MINIO
    # ==========================================
    try:

        client = Minio(
            MINIO_ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False
        )

        bucket_name = "bucket-predictions"

        if not client.bucket_exists(bucket_name):

            client.make_bucket(bucket_name)

        # ======================================
        # TO CSV
        # ======================================
        pdf = result_df.toPandas()

        csv_bytes = pdf.to_csv(
            index=False
        ).encode("utf-8")

        obj_name = (
            f"predictions/"
            f"batch_{epoch_id}_"
            f"{int(time.time())}.csv"
        )

        # ======================================
        # UPLOAD
        # ======================================
        client.put_object(
            bucket_name,
            obj_name,
            io.BytesIO(csv_bytes),
            length=len(csv_bytes),
            content_type="text/csv",
        )

        log.info(
            f"[Batch {epoch_id}] "
            f"{len(pdf)} predictions saved → "
            f"MinIO ({bucket_name}/{obj_name})"
        )

    except Exception as e:

        log.error(
            f"[MINIO ERROR] "
            f"Batch {epoch_id}: {e}"
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
