if __name__ == "__main__":
    # ==========================
    # 1. CÀI ĐẶT THÔNG SỐ (HYPERPARAMETERS)
    # ==========================
    seq_len = 10        # Nhìn lại 10 cây nến
    batch_size = 16
    hidden_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # ==========================
    # 2. XỬ LÝ DỮ LIỆU
    # ==========================
    # Giả sử bạn đã load df_raw từ csv
    # df_raw = pd.read_csv('000001.csv')
    
    # Mock data để code có thể chạy ngay
    data_str = """datetime,symbol,open,high,low,close,volume
2025-02-05 08:30:00,SSE:000001,3270.2,3271.0,3235.1,3239.8,100
2025-02-05 08:45:00,SSE:000001,3239.8,3245.6,3234.7,3235.3,120
2025-02-05 09:00:00,SSE:000001,3235.1,3237.9,3228.1,3237.5,110
2025-02-05 09:15:00,SSE:000001,3237.4,3241.3,3237.2,3239.5,130
2025-02-05 09:30:00,SSE:000001,3239.5,3245.0,3238.0,3242.0,150
2025-02-05 09:45:00,SSE:000001,3242.0,3250.0,3240.0,3248.0,200"""
    from io import StringIO
    df_raw = pd.read_csv(StringIO(data_str))

    # Áp dụng các hàm của bạn
    df_clean = preprocess_data(df_raw)
    df_ema = calculate_ema5(df_clean)

    # Chọn các cột để làm Features (Đầu vào)
    # Ta bỏ 'datetime', 'symbol', 'timestamp' ra khỏi đầu vào của model
    features_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA5']
    data_values = df_ema[features_cols].values 
    input_size = len(features_cols) # Sẽ là 6
    
    # Xác định cột mục tiêu (Dự đoán giá 'close' -> index là 3 trong features_cols)
    target_col_idx = 3

    # Scaler (Rất quan trọng)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    # ==========================
    # 3. TẠO DATASET & DATALOADER
    # ==========================
    X, y = create_sliding_windows(scaled_data, target_col_idx, seq_len)

    # Train/Test Split (Chia theo thứ tự thời gian, 80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Time-series không nên shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==========================
    # 4. KHỞI TẠO MODEL, LOSS, OPTIMIZER
    # ==========================
    model = HBLSTM(input_size=input_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss() # Hàm mất mát Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ==========================
    # 5. VÒNG LẶP TRAIN & TEST
    # ==========================
    print("Bắt đầu quá trình huấn luyện...")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            test_loss = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")