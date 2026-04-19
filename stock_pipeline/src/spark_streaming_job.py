import os
import sys
import pyspark
import json
import torch
import joblib
import pandas as pd
import numpy as np
import psycopg2

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# --- 0. KHÓA CHẶT MÔI TRƯỜNG PYSPARK VÀ JAVA VÀO VENV ---
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['SPARK_HOME'] = os.path.dirname(pyspark.__file__)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from minio import Minio

# Import Model của bạn (Đảm bảo thư viện utils/module có trong PYTHONPATH)
from module.model import HBLSTM
from module.data_processor import calculate_ema5

print("1. Đã import thư viện thành công!")

# --- 1. CẤU HÌNH THÔNG SỐ ---
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "realtime-stock-data"
# CHECKPOINT_LOCATION = "./checkpoint"
CHECKPOINT_LOCATION = "file://" + os.path.abspath("./checkpoint")

MINIO_ENDPOINT = "localhost:9005"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"

# Cấu hình PostgreSQL (Bạn nhớ sửa lại cho đúng với DB của bạn)
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stock_db"
DB_USER = "postgres"
DB_PASS = "postgres"

print("2. Đang khởi tạo Spark Session...")

# --- 2. KHỞI TẠO SPARK SESSION (Tự động thích ứng phiên bản) ---
# Lấy chính xác phiên bản PySpark đang chạy trong venv
spark_version = pyspark.__version__

# Spark 4.0+ mặc định dùng Scala 2.13, ngược lại dùng 2.12
scala_version = "2.13" if int(spark_version.split(".")[0]) >= 4 else "2.12"
kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}"

print(f"   -> Đang tải gói Kafka: {kafka_package}")

spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .config("spark.jars.packages", kafka_package) \
    .getOrCreate()
    
# Giảm bớt log rác của Spark để dễ nhìn terminal
spark.sparkContext.setLogLevel("WARN")

print("3. Khởi tạo Spark thành công! Đang tải Model từ MinIO...")

# --- 3. TẢI MODEL & SCALER TỪ MINIO ---
def load_artifacts():
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
    
    client.fget_object("bucket-models", "hblstm_model.pth", "tmp_model.pth")
    client.fget_object("bucket-models", "scaler.pkl", "tmp_scaler.pkl")
    client.fget_object("bucket-models", "model_config.json", "tmp_config.json")
    
    with open("tmp_config.json", "r") as f:
        config = json.load(f)
    
    scaler = joblib.load("tmp_scaler.pkl")
    
    model = HBLSTM(input_size=config['input_size'], hidden_size=config['hidden_size'])
    model.load_state_dict(torch.load("tmp_model.pth"))
    model.eval()
    
    return model, scaler, config

# Khởi tạo model trên Driver
model, scaler, config = load_artifacts()
print("   -> Đã tải Model thành công!")

# --- 4. HÀM XỬ LÝ THEO TỪNG BATCH (Dự đoán & Ghi Database) ---
def process_batch(df_batch, batch_id):
    # Nếu batch trống thì bỏ qua
    if df_batch.isEmpty():
        return
        
    # Chuyển đổi Spark DataFrame thành Pandas DataFrame để đưa vào PyTorch
    pdf = df_batch.toPandas()
    
    # 1. Tiền xử lý dữ liệu (Ví dụ tính EMA5 và scale)
    # Tùy thuộc vào cấu trúc dữ liệu Kafka mà bạn truyền vào, hãy sửa tên cột 'close' cho phù hợp
    if 'close' in pdf.columns:
        pdf['ema5'] = calculate_ema5(pdf['close'])
        
        # Tạo input tensor cho model (Ví dụ model nhận mảng 2D)
        features = pdf[['close', 'ema5']].fillna(0).values
        features_scaled = scaler.transform(features)
        
        # Reshape cho LSTM (batch_size, sequence_length, input_size)
        # Giả sử sequence_length = 1 cho dữ liệu realtime từng điểm
        inputs = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1)
        
        # 2. Dự đoán với Model PyTorch
        with torch.no_grad():
            predictions = model(inputs)
            
        pdf['predicted_price'] = predictions.numpy().flatten()
        
        # 3. Ghi kết quả vào PostgreSQL
        try:
            conn = psycopg2.connect(
                host=DB_HOST, port=DB_PORT, database=DB_NAME,
                user=DB_USER, password=DB_PASS
            )
            cursor = conn.cursor()
            
            for _, row in pdf.iterrows():
                # Giả sử bảng tên là 'stock_predictions'
                insert_query = """
                    INSERT INTO stock_predictions (symbol, timestamp, actual_price, predicted_price)
                    VALUES (%s, %s, %s, %s)
                """
                # Chú ý thay đổi tên cột 'symbol', 'timestamp' cho khớp với dữ liệu thật của bạn
                cursor.execute(insert_query, (row.get('symbol', 'UNK'), row.get('timestamp', None), row['close'], row['predicted_price']))
                
            conn.commit()
            cursor.close()
            conn.close()
            print(f"Batch {batch_id}: Đã xử lý và lưu {len(pdf)} bản ghi vào Database.")
            
        except Exception as e:
            print(f"Lỗi khi ghi Database ở Batch {batch_id}: {e}")

# --- 5. ĐỌC STREAM TỪ KAFKA ---
print("4. Bắt đầu lắng nghe luồng dữ liệu từ Kafka...")

# Định nghĩa Schema của cục JSON nhận từ Kafka
kafka_schema = StructType([
    StructField("symbol", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("close", FloatType(), True)
    # Thêm các cột khác nếu có
])

# Đọc luồng Kafka
df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Giải mã dữ liệu Value từ Kafka (dạng bytes) sang JSON
df_parsed = df_stream.selectExpr("CAST(value AS STRING) as json_str") \
    .select(from_json(col("json_str"), kafka_schema).alias("data")) \
    .select("data.*")

# --- 6. GHI STREAM (KÍCH HOẠT QUÁ TRÌNH) ---
query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", CHECKPOINT_LOCATION) \
    .start()

query.awaitTermination()