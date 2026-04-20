"""
stream_consumer.py
==================
Spark Structured Streaming: Kafka -> Tiền xử lý -> Dự đoán -> Incremental Update -> PostgreSQL + MinIO

Luồng dữ liệu:
  Kafka (raw OHLCV, field "date")
    -> parse & tính EMA5 (dựa trên close_history buffer)
    -> scale bằng scaler offline
    -> dự đoán close (dùng SEQ_LEN row liền trước từ seq_buffer)
    -> incremental update model với true label vừa đến
    -> lưu [predicted, actual] vào PostgreSQL
    -> tích lũy raw data mới, upload lên MinIO theo từng batch
  Khi tắt (Ctrl+C):
    -> lưu model đã cập nhật lên MinIO (cho lần retrain offline sau)
"""

import os
import sys
import json
import time
import logging
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import psycopg2
import pyspark
from minio import Minio

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType,
)

from module.model import HBLSTM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stream_consumer")

# ──────────────────────────────────────────────
# 0. KHÓA CHẶT MÔI TRƯỜNG PYSPARK / JAVA
# ──────────────────────────────────────────────
os.environ["JAVA_HOME"]             = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["SPARK_HOME"]            = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ──────────────────────────────────────────────
# 1. CẤU HÌNH THÔNG SỐ
# ──────────────────────────────────────────────
KAFKA_BROKER        = "localhost:9092"
KAFKA_TOPIC         = "realtime-stock-data"
CHECKPOINT_LOCATION = "file://" + os.path.abspath("./checkpoint")

MINIO_ENDPOINT   = "localhost:9005"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_MODELS    = "bucket-models"
BUCKET_DATA      = "bucket-data"

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stock_db"
DB_USER = "user"
DB_PASS = "password123"

ASSETS_DIR        = "./runtime_assets"
INCREMENTAL_LR    = 1e-4
TRIGGER_INTERVAL  = "5 seconds"
EMA_SPAN          = 5
# Số rows lịch sử cuối dùng warm-up tính EMA5 (không ảnh hưởng SEQ_LEN)
EMA_SEED_ROWS     = 30

os.makedirs(ASSETS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 2. KẾT NỐI MINIO & TẢI TÀI NGUYÊN
# ──────────────────────────────────────────────
log.info("Kết nối MinIO và tải tài nguyên...")
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def _dl(bucket: str, obj: str, dest: str) -> str:
    """Tải file từ MinIO về local, trả về đường dẫn local."""
    minio_client.fget_object(bucket, obj, dest)
    log.info(f"  ✓ {bucket}/{obj} -> {dest}")
    return dest

_dl(BUCKET_MODELS, "hblstm_model.pth",        f"{ASSETS_DIR}/hblstm_model.pth")
_dl(BUCKET_MODELS, "scaler.pkl",               f"{ASSETS_DIR}/scaler.pkl")
_dl(BUCKET_MODELS, "model_config.json",        f"{ASSETS_DIR}/model_config.json")
_dl(BUCKET_DATA,   "historical_data.parquet",  f"{ASSETS_DIR}/historical_data.parquet")

# ──────────────────────────────────────────────
# 3. LOAD CONFIG, SCALER, MODEL, HISTORY
# ──────────────────────────────────────────────
with open(f"{ASSETS_DIR}/model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

SEQ_LEN        = config["seq_len"]          # e.g. 3
INPUT_SIZE     = config["input_size"]       # e.g. 6
HIDDEN_SIZE    = config["hidden_size"]
FEATURES_COLS  = config["features_cols"]    # ['open','high','low','close','volume','EMA5']
TARGET_COL_IDX = config["target_col_idx"]   # e.g. 3 (close)

scaler = joblib.load(f"{ASSETS_DIR}/scaler.pkl")

device = torch.device("cpu")
model  = HBLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
model.load_state_dict(torch.load(f"{ASSETS_DIR}/hblstm_model.pth", map_location=device))
model.eval()
log.info("Model HBLSTM đã được tải thành công.")

# ── 3a. Lịch sử để khởi tạo các buffer ──────────────────────────────────────
df_history = pd.read_parquet(f"{ASSETS_DIR}/historical_data.parquet")

# ── 3b. Close-price buffer: warm-up để tính EMA5 liên tục ───────────────────
# Lấy EMA_SEED_ROWS dòng cuối (chưa scale) để pandas ewm có đủ "lịch sử"
close_history: deque = deque(
    df_history["close"].tail(EMA_SEED_ROWS).tolist(),
    maxlen=EMA_SEED_ROWS + 500,   # Giới hạn RAM nhưng đủ rộng
)

# ── 3c. Sequence buffer: các row đã scale để tạo input window cho model ──────
# FIX: seed đúng SEQ_LEN - 1 rows, để khi row đầu tiên đến và được append
#      vào buffer thì ta có đúng SEQ_LEN rows để predict row thứ hai.
#      Điều này phù hợp với logic: "lấy SEQ_LEN row trước để predict row hiện tại".
_hist_scaled = scaler.transform(df_history[FEATURES_COLS].values)
seq_buffer: deque = deque(
    _hist_scaled[-(SEQ_LEN - 1):].tolist() if SEQ_LEN > 1 else [],
    maxlen=SEQ_LEN * 10,
)

# ── 3d. Accumulator: tích lũy raw data mới, upload MinIO sau mỗi batch ───────
new_rows_accumulator: list = []

log.info(
    f"Buffer đã seed: close_history={len(close_history)} rows | "
    f"seq_buffer={len(seq_buffer)}/{SEQ_LEN - 1} rows (cần thêm 1 row để predict)."
)

# ──────────────────────────────────────────────
# 4. POSTGRESQL – KHỞI TẠO BẢNG
# ──────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
    )

def init_postgres():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_predictions (
            id              SERIAL PRIMARY KEY,
            trade_date      VARCHAR(30) UNIQUE,
            open_price      DOUBLE PRECISION,
            high_price      DOUBLE PRECISION,
            low_price       DOUBLE PRECISION,
            close_actual    DOUBLE PRECISION,
            close_predicted DOUBLE PRECISION,
            ema5            DOUBLE PRECISION,
            volume          DOUBLE PRECISION,
            inc_loss        DOUBLE PRECISION,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    log.info("PostgreSQL: bảng stock_predictions sẵn sàng.")

init_postgres()

# ──────────────────────────────────────────────
# 5. HÀM TIỆN ÍCH
# ──────────────────────────────────────────────

def compute_ema5(close_list: list) -> float:
    """
    Tính EMA5 từ chuỗi close price, trả về giá trị mới nhất.
    adjust=False để match cách tính pandas truyền thống.
    """
    s   = pd.Series(close_list, dtype=float)
    ema = s.ewm(span=EMA_SPAN, adjust=False).mean()
    return float(ema.iloc[-1])


def predict(x_seq: list) -> float:
    """
    Dự đoán giá close (đã scale) từ x_seq có shape (SEQ_LEN, INPUT_SIZE).
    Trả về giá trị scaled.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
        return float(model(x).item())


def inverse_close(scaled_val: float) -> float:
    """
    Inverse transform cho riêng cột close (TARGET_COL_IDX).
    Tạo dummy vector full-zero rồi điền vào đúng vị trí cột close.
    """
    dummy = np.zeros((1, INPUT_SIZE))
    dummy[0, TARGET_COL_IDX] = scaled_val
    return float(scaler.inverse_transform(dummy)[0, TARGET_COL_IDX])


# Optimizer & loss dùng chung cho mọi incremental update
inc_optimizer = torch.optim.Adam(model.parameters(), lr=INCREMENTAL_LR)
inc_criterion = nn.MSELoss()


def incremental_update(x_seq: list, y_true_scaled: float) -> float:
    """
    Fine-tune model 1 bước trên sample mới nhất.
    x_seq          : (SEQ_LEN, INPUT_SIZE) – window TRƯỚC row hiện tại
    y_true_scaled  : close thực tế đã scale của row hiện tại
    Trả về loss.
    """
    model.train()
    x = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    y = torch.tensor([[y_true_scaled]], dtype=torch.float32).to(device)

    inc_optimizer.zero_grad()
    loss = inc_criterion(model(x), y)
    loss.backward()
    inc_optimizer.step()
    model.eval()
    return float(loss.item())


def save_model_to_minio(tag: str = "updated"):
    """Serialize trọng số model hiện tại và upload lên MinIO."""
    local    = f"{ASSETS_DIR}/hblstm_model_{tag}.pth"
    obj_name = f"hblstm_model_{tag}.pth"
    torch.save(model.state_dict(), local)
    minio_client.fput_object(BUCKET_MODELS, obj_name, local)
    log.info(f"Model đã lưu lên MinIO: {BUCKET_MODELS}/{obj_name}")


def flush_new_data_to_minio():
    """Upload dữ liệu stream tích lũy trong batch hiện tại lên MinIO (parquet)."""
    if not new_rows_accumulator:
        return
    df_new   = pd.DataFrame(new_rows_accumulator)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    local    = f"{ASSETS_DIR}/stream_{ts}.parquet"
    obj_name = f"stream/stream_{ts}.parquet"
    df_new.to_parquet(local, index=False)
    minio_client.fput_object(BUCKET_DATA, obj_name, local)
    log.info(f"MinIO: {len(df_new)} rows mới -> {BUCKET_DATA}/{obj_name}")
    new_rows_accumulator.clear()


# ──────────────────────────────────────────────
# 6. FOREACHBATCH HANDLER
# ──────────────────────────────────────────────
def process_batch(batch_df, batch_id: int):
    """
    Được Spark gọi mỗi micro-batch.

    Với mỗi row (theo thứ tự thời gian):
      1. Tính EMA5 từ close_history buffer.
      2. Scale feature vector [open, high, low, close, volume, EMA5].
      3. Nếu seq_buffer đủ SEQ_LEN rows → predict close row hiện tại.
      4. Incremental update model với true label.
      5. Append scaled row hiện tại vào seq_buffer.
      6. Ghi kết quả vào PostgreSQL.
      7. Tích lũy raw row vào accumulator.
    Sau khi xử lý xong batch:
      8. Upload accumulated raw data lên MinIO.
    """
    # Khai báo global để sửa các biến trạng thái module-level
    global close_history, seq_buffer, new_rows_accumulator

    if batch_df.rdd.isEmpty():
        log.info(f"[Batch {batch_id}] Rỗng, bỏ qua.")
        return

    # ── FIX: sort theo đúng tên field "date" khớp với schema ─────────────────
    rows = batch_df.orderBy("date").collect()
    log.info(f"[Batch {batch_id}] Nhận {len(rows)} row(s).")

    conn = get_conn()
    cur  = conn.cursor()

    for row in rows:
        try:
            # ── FIX: đọc đúng tên field "date" (không phải "datetime") ────────
            date_str = str(row["date"])
            open_    = float(row["open"])
            high_    = float(row["high"])
            low_     = float(row["low"])
            close_   = float(row["close"])
            volume_  = float(row["volume"])

            # ── Bước 1: Cập nhật close_history & tính EMA5 ───────────────────
            close_history.append(close_)
            ema5 = compute_ema5(list(close_history))

            # ── Bước 2: Scale feature vector ─────────────────────────────────
            raw_vec      = np.array([[open_, high_, low_, close_, volume_, ema5]])
            scaled_vec   = scaler.transform(raw_vec)[0]      # shape: (INPUT_SIZE,)
            scaled_close = float(scaled_vec[TARGET_COL_IDX])

            # ── Bước 3 & 4: Predict + Incremental Update ─────────────────────
            close_predicted = None
            inc_loss        = None

            if len(seq_buffer) >= SEQ_LEN:
                # Lấy SEQ_LEN row cuối của buffer (TRƯỚC khi append row hiện tại)
                # → đây là input window để dự đoán close của row hiện tại
                x_seq = list(seq_buffer)[-SEQ_LEN:]         # (SEQ_LEN, INPUT_SIZE)

                pred_scaled     = predict(x_seq)
                close_predicted = inverse_close(pred_scaled)

                # Fine-tune với true label vừa quan sát được
                inc_loss = incremental_update(x_seq, scaled_close)

                log.info(
                    f"  [{date_str}] Actual={close_:.4f} | "
                    f"Predict={close_predicted:.4f} | "
                    f"EMA5={ema5:.4f} | Loss={inc_loss:.6f}"
                )
            else:
                log.warning(
                    f"  [{date_str}] Buffer {len(seq_buffer)}/{SEQ_LEN} – "
                    f"chưa đủ để dự đoán, đang warm-up..."
                )

            # ── Bước 5: Append row hiện tại vào seq_buffer ───────────────────
            seq_buffer.append(scaled_vec.tolist())

            # ── Bước 6: Ghi vào PostgreSQL ────────────────────────────────────
            cur.execute(
                """
                INSERT INTO stock_predictions
                    (trade_date, open_price, high_price, low_price,
                     close_actual, close_predicted, ema5, volume, inc_loss)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trade_date) DO UPDATE SET
                    close_actual    = EXCLUDED.close_actual,
                    close_predicted = EXCLUDED.close_predicted,
                    ema5            = EXCLUDED.ema5,
                    inc_loss        = EXCLUDED.inc_loss,
                    created_at      = NOW()
                """,
                (
                    date_str, open_, high_, low_,
                    close_, close_predicted, ema5, volume_, inc_loss,
                ),
            )

            # ── Bước 7: Tích lũy raw data ─────────────────────────────────────
            new_rows_accumulator.append({
                "date":   date_str,
                "open":   open_,
                "high":   high_,
                "low":    low_,
                "close":  close_,
                "volume": volume_,
                "EMA5":   ema5,
            })

        except Exception as exc:
            log.error(f"  Lỗi xử lý row {row}: {exc}", exc_info=True)

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"[Batch {batch_id}] Đã commit {len(rows)} row(s) vào PostgreSQL.")

    # ── Bước 8: Upload raw data batch này lên MinIO ────────────────────────────
    flush_new_data_to_minio()


# ──────────────────────────────────────────────
# 7. KHỞI TẠO SPARK SESSION
# ──────────────────────────────────────────────
log.info("Khởi tạo Spark Session...")
spark_version = pyspark.__version__
# Spark 4.x dùng Scala 2.13, Spark 3.x dùng Scala 2.12
scala_version = "2.13" if int(spark_version.split(".")[0]) >= 4 else "2.12"
kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}"
log.info(f"Kafka package: {kafka_package}")

spark = (
    SparkSession.builder
    .appName("StockPricePrediction_Streaming")
    .config("spark.jars.packages", kafka_package)
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
log.info("Spark Session sẵn sàng.")

# ──────────────────────────────────────────────
# 8. SCHEMA KAFKA MESSAGE
# ──────────────────────────────────────────────
# FIX: Tên field "date" (khớp với Producer gửi và process_batch đọc)
# Không dùng "datetime" – tránh nhầm lẫn với kiểu dữ liệu TimestampType
kafka_schema = StructType([
    StructField("date",   StringType(), True),   # FIX: "date" thay vì "datetime"
    StructField("open",   DoubleType(), True),
    StructField("high",   DoubleType(), True),
    StructField("low",    DoubleType(), True),
    StructField("close",  DoubleType(), True),
    StructField("volume", DoubleType(), True),
])

# ──────────────────────────────────────────────
# 9. ĐỌC STREAM TỪ KAFKA & PARSE JSON
# ──────────────────────────────────────────────
raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", KAFKA_TOPIC)
    # "latest": chỉ đọc tin nhắn MỚI kể từ khi consumer khởi động.
    # Đổi thành "earliest" để replay toàn bộ topic từ đầu.
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

# Kafka value (binary) -> string -> parse JSON theo kafka_schema
parsed_stream = (
    raw_stream
    .select(
        from_json(col("value").cast("string"), kafka_schema).alias("data")
    )
    .select("data.*")
    # Lọc bỏ row parse lỗi (close = null)
    .filter(col("close").isNotNull())
)

# ──────────────────────────────────────────────
# 10. KHỞI ĐỘNG STREAMING QUERY
# ──────────────────────────────────────────────
query = (
    parsed_stream.writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", CHECKPOINT_LOCATION)
    .trigger(processingTime=TRIGGER_INTERVAL)
    .start()
)

log.info("=" * 60)
log.info(f"Streaming bắt đầu! Topic: {KAFKA_TOPIC} | Trigger: {TRIGGER_INTERVAL}")
log.info("Nhấn Ctrl+C để dừng và lưu model.")
log.info("=" * 60)

try:
    query.awaitTermination()

except KeyboardInterrupt:
    log.info("Đang dừng streaming...")
    query.stop()

    # Lưu model đã được incremental update lên MinIO
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model_to_minio(tag=f"incremental_{ts}")

    # Flush dữ liệu còn trong accumulator (nếu batch cuối chưa kịp upload)
    flush_new_data_to_minio()

    log.info("✅ Đã dừng sạch. Model + dữ liệu mới đã lưu lên MinIO.")

except Exception as exc:
    log.error(f"Lỗi nghiêm trọng: {exc}", exc_info=True)
    query.stop()
    raise
