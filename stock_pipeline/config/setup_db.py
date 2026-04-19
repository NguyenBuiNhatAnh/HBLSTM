import psycopg2

# Thông tin kết nối lấy từ docker-compose.yml của bạn
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "stock_db"     # Sửa lại thành tên DB bạn đã setup, mặc định thường là postgres hoặc stock_db
DB_USER = "user"     # Tài khoản mặc định
DB_PASS = "password123"     # Mật khẩu mặc định

# Câu lệnh SQL tạo bảng
CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS stock_predictions (
    id SERIAL PRIMARY KEY,
    record_date VARCHAR(50),
    actual_open FLOAT,
    actual_high FLOAT,
    actual_low FLOAT,
    actual_close FLOAT,
    volume FLOAT,
    predicted_close FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

try:
    # 1. Kết nối tới PostgreSQL
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    
    # 2. Thực thi lệnh tạo bảng
    cursor.execute(CREATE_TABLE_QUERY)
    conn.commit()
    
    print(" Đã tạo bảng 'stock_predictions' thành công trong PostgreSQL!")
    
except Exception as e:
    print(f" Lỗi kết nối hoặc tạo bảng DB: {e}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()