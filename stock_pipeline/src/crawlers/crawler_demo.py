import pandas as pd
import json
import time
from kafka import KafkaProducer

# 1. Cấu hình Kafka Producer
KAFKA_BROKER = 'localhost:9092'
TOPIC_NAME = 'realtime-stock-data'

try:
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print(f"Đã kết nối thành công tới Kafka Broker: {KAFKA_BROKER}")
except Exception as e:
    print(f"Lỗi kết nối Kafka: {e}")
    exit(1)

# 2. Đọc file CSV để mô phỏng streaming
csv_file = '000001.csv'
print(f"Đang đọc dữ liệu từ {csv_file} để mô phỏng realtime...")
df_raw = pd.read_csv('000001.csv')


# Lấy 100 dòng cuối cùng (những ngày gần nhất) để test stream
# Bạn có thể bỏ .tail(100) đi nếu muốn stream toàn bộ file
stream_data = df_raw.tail(100).to_dict(orient='records')

# 3. Bắn dữ liệu vào Kafka từng giây một
print(f"Bắt đầu đẩy dữ liệu vào topic '{TOPIC_NAME}'...")
print("-" * 50)

for row in stream_data:
    # row là 1 dictionary chứa: date, open, high, low, close, volume...
    producer.send(TOPIC_NAME, value=row)
    producer.flush() # Đảm bảo tin nhắn được gửi đi ngay lập tức
    
    # In ra màn hình để quan sát
    print(f"[ĐÃ GỬI] Ngày: {row.get('date', 'N/A')} | Giá Close: {row.get('close', 0)}")
    
    # Tạm dừng 1.5 giây trước khi gửi dòng tiếp theo (Mô phỏng Realtime)
    time.sleep(1.5)

print("Đã hoàn tất việc đẩy dữ liệu mô phỏng!")