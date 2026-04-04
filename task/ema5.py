import pandas as pd

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