from pyspark.sql import SparkSession

# 1. Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("KafkaTriggerFunction") \
    .getOrCreate()

# Giảm bớt log rác trên màn hình console
spark.sparkContext.setLogLevel("WARN")

# 2. Đọc luồng dữ liệu (Stream) từ Kafka
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ten_topic_cua_ban") \
    .option("startingOffsets", "latest") \
    .load()

# Dữ liệu Kafka trả về ở dạng Binary, cần chuyển cột 'value' sang String để đọc chữ
string_df = kafka_df.selectExpr("CAST(value AS STRING)")

# 3. Định nghĩa hàm tùy chỉnh CỦA BẠN
# Hàm này sẽ tự động chạy mỗi khi nhận được dữ liệu mới
def my_custom_function(batch_df, batch_id):
    print(f"\n--- Đang nhận dữ liệu mới ở Batch ID: {batch_id} ---")
    
    # Collect dữ liệu trong batch hiện tại về dạng list để xử lý
    # Lưu ý: Chỉ dùng collect() nếu lượng dữ liệu trong 1 batch nhỏ/vừa phải
    records = batch_df.collect()
    
    if not records:
        return # Nếu không có dữ liệu thì bỏ qua
        
    for row in records:
        message = row['value']
        print(f"Nội dung message: {message}")
        
        # ---> BẠN GỌI CÁC HÀM XỬ LÝ LOGIC CỦA BẠN Ở ĐÂY <---
        # Ví dụ: send_to_database(message), trigger_api(message), v.v.

# 4. Kích hoạt luồng chạy liên tục
query = string_df \
    .writeStream \
    .outputMode("append") \
    .foreachBatch(my_custom_function) \
    .start()

# Giữ cho Spark chạy liên tục để chờ message
query.awaitTermination()