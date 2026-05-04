import pandas as pd

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
    df_clean['timestamp'] = df_clean['datetime'].apply(lambda x: int(x.timestamp()))

    # (Tùy chọn) Reset lại index của bảng cho gọn gàng sau khi xóa dòng và sắp xếp
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def calculate_ema5(df):
    """
    Hàm tính toán đường trung bình động hàm mũ EMA5.
    Đầu vào: df (DataFrame đã được tiền xử lý và sắp xếp theo thời gian)
    """
    df_ema = df.copy()
    
    # Tính EMA5 cho cột 'close'
    # span=5: Chu kỳ 5 nến
    # adjust=False: Sử dụng công thức đệ quy tiêu chuẩn (giống hệ thống TradingView)
    df_ema['EMA5'] = df_ema['close'].ewm(span=5, adjust=False).mean()
    
    return df_ema