import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Giữ lại các class và hàm bạn đã viết ở trên:
# HLSTMCell, HBLSTM, preprocess_data, calculate_ema5

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Ép kiểu về float32 để tương thích tốt với PyTorch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sliding_windows(data, target_col_idx, seq_len):
    """
    Tạo các chuỗi dữ liệu đầu vào (X) và nhãn (y) theo dạng cửa sổ trượt.
    Dự đoán giá trị của bước thời gian tiếp theo.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :]) # Lấy seq_len dòng liên tiếp
        y.append(data[i + seq_len, target_col_idx]) # Dự đoán mục tiêu ở dòng tiếp theo
    return np.array(X), np.array(y)