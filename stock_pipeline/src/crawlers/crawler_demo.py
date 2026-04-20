"""
kafka_producer.py
=================
Đọc file CSV và đẩy từng dòng vào Kafka topic theo thời gian thực.

Chỉ gửi các cột: date, open, high, low, close, volume
(đã chuẩn hóa tên cột để khớp với schema bên Consumer)
"""

import json
import time
import logging

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kafka_producer")

# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
KAFKA_BROKER    = "localhost:9092"
TOPIC_NAME      = "realtime-stock-data"
CSV_FILE        = "000001.csv"
ROWS_TO_STREAM  = 100      # Số dòng cuối cùng sẽ được stream; None = toàn bộ
SEND_INTERVAL   = 1.5      # Giây giữa mỗi lần gửi

# ──────────────────────────────────────────────
# CÁC TÊN CỘT CÓ THỂ GẶP TRONG CSV
# Key = tên cột CSV (thường gặp), Value = tên chuẩn gửi lên Kafka
# ──────────────────────────────────────────────
COLUMN_RENAME_MAP = {
    # date variants
    "Date":     "date",
    "date":     "date",
    "Datetime": "date",
    "datetime": "date",
    "Time":     "date",
    "time":     "date",
    # OHLCV variants
    "Open":   "open",
    "High":   "high",
    "Low":    "low",
    "Close":  "close",
    "Volume": "volume",
    "open":   "open",
    "high":   "high",
    "low":    "low",
    "close":  "close",
    "volume": "volume",
}

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def load_and_prepare(csv_path: str, n_rows=None) -> list[dict]:
    """
    Đọc CSV, chuẩn hóa tên cột, chọn đúng cột cần thiết.
    Trả về list of dict để đẩy vào Kafka.
    """
    df = pd.read_csv(csv_path)
    log.info(f"Đã đọc {len(df)} dòng từ {csv_path}. Cột gốc: {df.columns.tolist()}")

    # Rename các cột theo map
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Kiểm tra cột bắt buộc
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Thiếu cột trong CSV: {missing}. "
            f"Cột hiện có sau rename: {df.columns.tolist()}"
        )

    # Chỉ giữ các cột cần thiết
    df = df[REQUIRED_COLS].copy()

    # Đảm bảo kiểu dữ liệu đúng
    df["date"]   = df["date"].astype(str)
    df["open"]   = pd.to_numeric(df["open"],   errors="coerce")
    df["high"]   = pd.to_numeric(df["high"],   errors="coerce")
    df["low"]    = pd.to_numeric(df["low"],    errors="coerce")
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Bỏ dòng có giá trị NaN
    before = len(df)
    df.dropna(inplace=True)
    if len(df) < before:
        log.warning(f"Đã loại bỏ {before - len(df)} dòng có giá trị NaN.")

    # Lấy N dòng cuối nếu cần
    if n_rows is not None:
        df = df.tail(n_rows)

    log.info(f"Sẽ stream {len(df)} dòng.")
    return df.to_dict(orient="records")


def main():
    # ── Kết nối Kafka ─────────────────────────────────────────────────────────
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Đảm bảo tin nhắn được gửi và confirm
            acks="all",
            retries=3,
        )
        log.info(f"Đã kết nối Kafka Broker: {KAFKA_BROKER}")
    except Exception as exc:
        log.error(f"Không thể kết nối Kafka: {exc}")
        raise SystemExit(1) from exc

    # ── Chuẩn bị dữ liệu ─────────────────────────────────────────────────────
    records = load_and_prepare(CSV_FILE, n_rows=ROWS_TO_STREAM)

    # ── Đẩy từng dòng ────────────────────────────────────────────────────────
    log.info(f"Bắt đầu đẩy dữ liệu vào topic '{TOPIC_NAME}'...")
    log.info("-" * 60)

    for idx, row in enumerate(records, start=1):
        msg = {
            "date":   str(row.get("datetime", "")),   # rename datetime -> date
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": float(row["volume"]),
            # bỏ "symbol" – consumer không cần
        }
        producer.send(TOPIC_NAME, value=msg)
        producer.flush()  # Gửi ngay lập tức, không buffer

        log.info(
            f"[{idx:>3}/{len(records)}] "
            f"Ngày: {row['date']} | "
            f"Open={row['open']:.4f} | "
            f"High={row['high']:.4f} | "
            f"Low={row['low']:.4f} | "
            f"Close={row['close']:.4f} | "
            f"Volume={row['volume']:.0f}"
        )

        if idx < len(records):
            time.sleep(SEND_INTERVAL)

    log.info("✅ Đã hoàn tất đẩy toàn bộ dữ liệu mô phỏng!")
    producer.close()


if __name__ == "__main__":
    main()
