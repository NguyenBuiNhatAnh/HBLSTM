import torch
import torch.nn as nn


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
        u = torch.sigmoid(u)   # update gate (🔥 mới)

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
# BIDIRECTIONAL HLSTM
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

        # ===== Forward direction =====
        h_f = torch.zeros(batch, self.hidden_size, device=device)
        c_f = torch.zeros_like(h_f)
        k_f = torch.zeros_like(h_f)

        outputs_f = []

        for t in range(seq_len):
            h_f, c_f, k_f = self.fw(x[:, t, :], h_f, c_f, k_f)
            outputs_f.append(h_f.unsqueeze(1))

        outputs_f = torch.cat(outputs_f, dim=1)

        # ===== Backward direction =====
        h_b = torch.zeros(batch, self.hidden_size, device=device)
        c_b = torch.zeros_like(h_b)
        k_b = torch.zeros_like(h_b)

        outputs_b = []

        for t in reversed(range(seq_len)):
            h_b, c_b, k_b = self.bw(x[:, t, :], h_b, c_b, k_b)
            outputs_b.insert(0, h_b.unsqueeze(1))

        outputs_b = torch.cat(outputs_b, dim=1)

        # ===== Concatenate =====
        out = torch.cat([outputs_f, outputs_b], dim=2)

        # ===== Many-to-one (lấy timestep cuối) =====
        out = out[:, -1, :]

        return self.fc(out)


# =========================
# TEST MODEL
# =========================
if __name__ == "__main__":
    batch = 4
    seq_len = 10
    input_size = 5
    hidden_size = 16

    x = torch.randn(batch, seq_len, input_size)

    model = HBLSTM(input_size, hidden_size)

    out = model(x)

    print("Output shape:", out.shape)  # (batch, 1)