import pandas as pd
import json
import time
from kafka import KafkaProducer

def preprocess_data(df):
    """
    Hàm tiền xử lý dữ liệu theo 4 bước yêu cầu.
    Đầu vào: df (Pandas DataFrame chứa dữ liệu gốc)
    """
    # Tạo một bản sao để không làm ảnh hưởng đến dữ liệu gốc
    df_clean = df.copy()

    # a. Remove null values (Xóa các dòng chứa giá trị rỗng/NA)
    df_clean = df_clean.dropna()

    # b. Remove duplicate instances (Xóa các dòng trùng lặp)
    # Khuyến nghị: Chỉ cần trùng lặp thời gian và mã cổ phiếu là tính là trùng
    df_clean = df_clean.drop_duplicates(subset=['datetime', 'symbol'])

    # Bước đệm: Ép kiểu cột datetime từ String (chuỗi) sang dạng Datetime của Pandas
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])

    # c. Ensure the order of data (Sắp xếp dữ liệu theo thứ tự thời gian tăng dần)
    df_clean = df_clean.sort_values(by='datetime', ascending=True)

    # d. Convert string date-time to numerical timestamp 
    # (Chuyển đổi thời gian thành con số dạng Unix timestamp - tính bằng giây)
    df_clean['timestamp'] = df_clean['datetime'].astype('int64') // 10**9

    # (Tùy chọn) Reset lại index của bảng cho gọn gàng sau khi xóa dòng và sắp xếp
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def calculate_ema5(data):
    import pandas as pd
    # Nếu data là Series (chuỗi giá trị), tính trực tiếp
    # Nếu data là DataFrame, lấy cột 'close'
    if isinstance(data, pd.DataFrame):
        return data['close'].ewm(span=5, adjust=False).mean()
    else:
        # Giả định data đã là chuỗi số (Series) của cột close
        return data.ewm(span=5, adjust=False).mean()

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

# Áp dụng các hàm tiền xử lý
df_clean = preprocess_data(df_raw)
df_ema = calculate_ema5(df_clean)

# Lấy 100 dòng cuối cùng (những ngày gần nhất) để test stream
# Bạn có thể bỏ .tail(100) đi nếu muốn stream toàn bộ file
stream_data = df_ema.tail(100).to_dict(orient='records')

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