import pandas as pd
from ..data.data_preprocessing import calculate_ema5, preprocess_data
from sklearn.preprocessing import MinMaxScaler
from ..data.dataset import TimeSeriesDataset, create_sliding_windows
from torch.utils.data import DataLoader
from models import LinearModel, DTModel, KNNModel, HBLSTMModel


# Cấu hình đường dẫn
data_paths = [
    {'symbol': 'DJI',     'path': '../data/stock_data/DJI.csv'},
    {'symbol': 'SPX',     'path': '../data/stock_data/SPX.csv'},
    {'symbol': 'NDX',     'path': '../data/stock_data/NDX.csv'},
    {'symbol': 'DXY',     'path': '../data/stock_data/DXY.csv'},
    {'symbol': 'NI225',   'path': '../data/stock_data/NI225.csv'},
    {'symbol': '000001',  'path': '../data/stock_data/000001.csv'},
    {'symbol': 'NIFTY',   'path': '../data/stock_data/NIFTY.csv'},
    {'symbol': 'UKX',     'path': '../data/stock_data/UKX.csv'},
    {'symbol': 'DEU40',   'path': '../data/stock_data/DEU40'},
]



for index in data_paths:
    
    
    print(f"Đang tải dữ liệu {index['symbol']}")
    df_raw = pd.read_csv(index['path'])
    
    # ==========================
    # 1. XỬ LÝ DỮ LIỆU & CHUẨN HÓA
    # ==========================
    
    # Tiền xử lý dữ liệu
    df_clean = preprocess_data(df_raw)
    # Tính ema5
    df_ema5 = calculate_ema5(df_clean)
    
    # Chọn các cột để làm Features (Đầu vào)
    features_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'EMA5']
    data_values = df_ema5[features_cols].values 
    input_size = len(features_cols) # Sẽ là 7
    
    # Xác định cột mục tiêu (Dự đoán giá 'close' -> index là 3)
    target_col_idx = 3
    
    # Xác định vị trí cắt mảng 2D thành Train/Test (80%)
    split_idx_raw = int(len(data_values) * 0.8)
    # Khởi tạo Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # BƯỚC QUAN TRỌNG: Chỉ dùng hàm fit() để dò tìm Min/Max trên tập Train
    scaler.fit(data_values[:split_idx_raw])
    # Sau đó mới áp dụng transform() để chuẩn hóa cho toàn bộ mảng dữ liệu
    scaled_data = scaler.transform(data_values)
    
    # ==========================
    # 2. TẠO DATASET & DATALOADER
    # ==========================
    
    X, y = create_sliding_windows(scaled_data, target_col_idx, 3)
    
    # Train/Test Split (Chia theo thứ tự thời gian, 80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Mẹo: Bật shuffle=True ở tập Train sẽ giúp mô hình hội tụ tốt hơn!
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    
        
    
    
        