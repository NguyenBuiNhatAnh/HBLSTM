# models/__init__.py

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import joblib
from ..data.dataset import TimeSeriesDataset
from torch.utils.data import DataLoader


# ========================================
# BASE CLASS
# ========================================
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y): pass
    
    @abstractmethod
    def predict(self, X): pass
    
    @abstractmethod
    def save(self, path): pass
    
    @abstractmethod
    def load(self, path): pass


# ========================================
# 1. LINEAR REGRESSION
# ========================================
class LinearModel(BaseModel):
    def __init__(self):
        self.model = None
    
    def train(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)


# ========================================
# 2. DECISION TREE
# ========================================
class DTModel(BaseModel):
    def __init__(self, max_depth=10):
        self.model = None
        self.max_depth = max_depth
    
    def train(self, X, y):
        from sklearn.tree import DecisionTreeRegressor
        self.model = DecisionTreeRegressor(max_depth=self.max_depth)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)


# ========================================
# 3. KNN
# ========================================
class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        self.model = None
        self.n_neighbors = n_neighbors
    
    def train(self, X, y):
        from sklearn.neighbors import KNeighborsRegressor
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)


# ========================================
# 4. HBLSTM
# ========================================

# ---- 4.1 HLSTM Cell ----
class HLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2h = nn.Linear(input_size, 5 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 5 * hidden_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, x, h, c, k):
        gates = self.x2h(x) + self.h2h(h)
        i, f, o, g, u = gates.chunk(5, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.sigmoid(u)
        g = torch.tanh(g)

        k = u * g + (1 - u) * k
        c = f * c + i * k
        h = o * torch.tanh(c)

        return h, c, k


# ---- 4.2 Network ----
class HBLSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.fw = HLSTMCell(input_size, hidden_size)
        self.bw = HLSTMCell(input_size, hidden_size)

        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        device = x.device

        # init states
        h_f = torch.zeros(batch, self.hidden_size, device=device)
        c_f = torch.zeros_like(h_f)
        k_f = torch.zeros_like(h_f)

        h_b = torch.zeros(batch, self.hidden_size, device=device)
        c_b = torch.zeros_like(h_b)
        k_b = torch.zeros_like(h_b)

        # forward
        for t in range(seq_len):
            h_f, c_f, k_f = self.fw(x[:, t, :], h_f, c_f, k_f)

        # backward
        for t in reversed(range(seq_len)):
            h_b, c_b, k_b = self.bw(x[:, t, :], h_b, c_b, k_b)

        out = torch.cat([h_f, h_b], dim=1)
        return self.fc(out)


# ---- 4.3 Wrapper ----
class HBLSTMModel(BaseModel):
    def __init__(self, input_size=8, hidden_size=64, seq_len=10, lr=1e-4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lr = lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ✅ FIX: dùng đúng network
        self.model = HBLSTMNetwork(input_size, hidden_size).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, X, y, epochs=50, batch_size=16):
        assert len(X.shape) == 3, "X must be (samples, seq_len, input_size)"

        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    def predict(self, X):
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy().flatten()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.seq_len = checkpoint["seq_len"]

        self.model = HBLSTMNetwork(self.input_size, self.hidden_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def incremental_update(self, X_new, y_new, lr=None):
        self.model.train()

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        X_tensor = torch.FloatTensor(X_new).to(self.device)
        y_tensor = torch.FloatTensor(y_new).to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor.unsqueeze(1))

        loss.backward()
        self.optimizer.step()

        return loss.item()


# ========================================
# EXPORT
# ========================================
__all__ = [
    'LinearModel',
    'DTModel',
    'KNNModel',
    'HBLSTMModel'
]