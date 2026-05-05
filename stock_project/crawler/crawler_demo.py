import pandas as pd
import time
import json
from kafka import KafkaProducer

# =========================
# 1. CONFIG
# =========================
symbols = [
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

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode(),
    key_serializer=lambda k: k.encode()
)


# =========================
# 2. LOAD + MERGE DATA
# =========================
dfs = []

for symbol in symbols:
    df = pd.read_csv(symbol['path'])
    df["symbol"] = symbol['symbol']
    dfs.append(df)

# Gộp tất cả
full_df = pd.concat(dfs)

# Sort theo thời gian (QUAN TRỌNG)
full_df = full_df.sort_values("datetime").reset_index(drop=True)

print("Tổng số record:", len(full_df))

# =========================
# 3. STREAM (REPLAY)
# =========================
for i, row in full_df.iterrows():

    data = {
        "symbol": row["symbol"],
        "datetime": str(row["datetime"]),  # giữ string cho an toàn
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"])
    }

    producer.send(
        topic="stock-price",
        key=row["symbol"],  # 🔥 QUAN TRỌNG (partition theo symbol)
        value=data
    )

    print(f"Sent: {data['symbol']} | {data['datetime']}")

    time.sleep(1)  # giả lập realtime (có thể chỉnh 0.5s, 2s tùy bạn)