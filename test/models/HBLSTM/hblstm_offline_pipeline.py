import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib


# =========================
# HLSTM CELL
# =========================
class HLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Input -> gates
        self.x2h = nn.Linear(input_size, 5 * hidden_size, bias=True)

        # Hidden -> gates (không bias để đúng paper hơn)
        self.h2h = nn.Linear(hidden_size, 5 * hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev, k_prev):
        """
        x: (batch, input_size)
        h_prev: hidden state (batch, hidden)
        c_prev: cell state (batch, hidden)
        k_prev: candidate memory K' (batch, hidden)
        """

        # ===== Gates computation =====
        gates = self.x2h(x) + self.h2h(h_prev)

        # Split thành 5 phần
        i, f, o, g, u = gates.chunk(5, dim=1)

        # ===== Activation =====
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        u = torch.sigmoid(u)   # update gate (mới)

        g = torch.tanh(g)      # candidate

        # ===== Hybrid candidate memory =====
        # k_t = u_t * g_t + (1 - u_t) * k_{t-1}
        k = u * g + (1 - u) * k_prev

        # ===== Cell state =====
        # c_t = f_t * c_{t-1} + i_t * k_t
        c = f * c_prev + i * k

        # ===== Hidden state =====
        h = o * torch.tanh(c)

        return h, c, k


# =========================
# BIDIRECTIONAL HBLSTM (Đã sửa & Tối ưu)
# =========================
class HBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.fw = HLSTMCell(input_size, hidden_size)
        self.bw = HLSTMCell(input_size, hidden_size)

        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        batch, seq_len, _ = x.shape
        device = x.device

        # ===== Khởi tạo state =====
        h_f = torch.zeros(batch, self.hidden_size, device=device)
        c_f = torch.zeros_like(h_f)
        k_f = torch.zeros_like(h_f)

        h_b = torch.zeros(batch, self.hidden_size, device=device)
        c_b = torch.zeros_like(h_b)
        k_b = torch.zeros_like(h_b)

        # ===== Forward direction =====
        # Chạy từ t=0 đến t=seq_len-1
        for t in range(seq_len):
            h_f, c_f, k_f = self.fw(x[:, t, :], h_f, c_f, k_f)
            # Sau khi hết vòng for, h_f chứa thông tin của cả chuỗi (chiều xuôi)

        # ===== Backward direction =====
        # Chạy ngược từ t=seq_len-1 về t=0
        for t in reversed(range(seq_len)):
            h_b, c_b, k_b = self.bw(x[:, t, :], h_b, c_b, k_b)
            # Sau khi hết vòng for, h_b chứa thông tin của cả chuỗi (chiều ngược)

        # ===== Many-to-one Concatenate =====
        # Ghép 2 hidden state tổng hợp của 2 chiều lại với nhau
        out = torch.cat([h_f, h_b], dim=1)  # shape: (batch, 2 * hidden_size)

        # ===== Final Prediction =====
        return self.fc(out)  # shape: (batch, 1)
    

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

def train_model(model, train_loader, criterion, optimizer, device):
    model.train() # Bật chế độ training
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(batch_X)
        
        # 3. Tính loss (outputs shape: [batch, 1], batch_y shape: [batch])
        # Cần unsqueeze batch_y thành [batch, 1] để cùng shape với outputs
        loss = criterion(outputs, batch_y.unsqueeze(1))
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Cập nhật trọng số
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval() # Bật chế độ evaluation (tắt Dropout, BatchNorm nếu có)
    total_loss = 0.0
    
    # Tắt tính toán gradient để tiết kiệm bộ nhớ và chạy nhanh hơn
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

# ==========================
# 1. CÀI ĐẶT THÔNG SỐ (HYPERPARAMETERS)
# ==========================
seq_len = 3         # Đã sửa lại comment: Nhìn lại 3 cây nến
batch_size = 16
hidden_size = 32
num_epochs = 50
learning_rate = 0.001
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Đang chạy trên thiết bị: {device}")

# ==========================
# 2. XỬ LÝ DỮ LIỆU & CHUẨN HÓA (ĐÃ FIX DATA LEAKAGE)
# ==========================

df_raw = pd.read_csv('000001.csv')

# Áp dụng các hàm tiền xử lý
df_clean = preprocess_data(df_raw)
df_ema = calculate_ema5(df_clean)

# Chọn các cột để làm Features (Đầu vào)
features_cols = ['open', 'high', 'low', 'close', 'volume', 'EMA5']
data_values = df_ema[features_cols].values 
input_size = len(features_cols) # Sẽ là 6
    
# Xác định cột mục tiêu (Dự đoán giá 'close' -> index là 3)
target_col_idx = 3

# --- BẮT ĐẦU SỬA DATA LEAKAGE ---
# Xác định vị trí cắt mảng 2D thành Train/Test (80%)
split_idx_raw = int(len(data_values) * 0.8)

# Khởi tạo Scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# BƯỚC QUAN TRỌNG: Chỉ dùng hàm fit() để dò tìm Min/Max trên tập Train
scaler.fit(data_values[:split_idx_raw])

# Sau đó mới áp dụng transform() để chuẩn hóa cho toàn bộ mảng dữ liệu
scaled_data = scaler.transform(data_values)
# --- KẾT THÚC SỬA DATA LEAKAGE ---

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

# Mẹo: Với chuỗi thời gian đã chia Window xong thành các khối độc lập, 
# bật shuffle=True ở tập Train sẽ giúp mô hình hội tụ tốt hơn!
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
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



# Lưu model (bạn đã làm)
torch.save(model.state_dict(), 'hblstm_model.pth')
# Lưu scaler (BẮT BUỘC)
joblib.dump(scaler, 'scaler.pkl')

