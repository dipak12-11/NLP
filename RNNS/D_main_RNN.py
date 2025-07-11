
import numpy as np

class WordRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # Removed W_x (embedding layer) — GloVe gives us the input vectors already
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_y = np.random.randn(hidden_size, output_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x_seq, h_prev):
        """
        x_seq: List of GloVe vectors, each shape (embedding_dim,)
        h_prev: shape (hidden_size, 1)
        """
        hs, ps = {}, {}
        hs[-1] = h_prev

        for t in range(len(x_seq)):
            x_t = x_seq[t].reshape(-1, 1)  # Already embedded
            h_next = np.tanh(np.dot(self.W_h, hs[t - 1]) + np.dot(x_t.T, self.b_h.T).T)
            y_hat = np.dot(self.W_y.T, h_next) + self.b_y
            p = np.exp(y_hat - np.max(y_hat)) / np.sum(np.exp(y_hat - np.max(y_hat)))
            hs[t] = h_next
            ps[t] = p
        return hs, ps

    def backward(self, x_seq, y_target, hs, ps):
        dWhh = np.zeros_like(self.W_h)
        dWhy = np.zeros_like(self.W_y)
        dbh = np.zeros_like(self.b_h)
        dby = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(hs[0])
        loss = 0

        t = len(x_seq) - 1
        dy = np.copy(ps[t])
        target_idx = y_target if isinstance(y_target, int) else y_target[0]
        dy[target_idx] -= 1
        loss += -np.log(ps[t][target_idx] + 1e-9)

        dWhy += np.dot(hs[t], dy.T)
        dby += dy
        dh = np.dot(self.W_y, dy) + dh_next

        for t in reversed(range(len(x_seq))):
            h = hs[t]
            h_prev = hs[t - 1]
            dh_raw = (1 - h * h) * dh
            dbh += dh_raw
            dWhh += np.dot(dh_raw, h_prev.T)
            dh = np.dot(self.W_h.T, dh_raw)

        # Clip
        for dparam in [dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return loss, dWhh, dWhy, dbh, dby
