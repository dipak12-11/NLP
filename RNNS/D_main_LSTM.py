import numpy as np
class WordLSTM:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Embedding matrix
        self.E = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Input gate params
        self.W_ix = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_ih = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_i = np.zeros((hidden_dim, 1))

        # Forget gate params
        self.W_fx = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_fh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_f = np.zeros((hidden_dim, 1))

        # Output gate params
        self.W_ox = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_oh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_o = np.zeros((hidden_dim, 1))

        # Cell gate params
        self.W_cx = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_ch = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_c = np.zeros((hidden_dim, 1))

        # Output layer
        self.W_y = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_y = np.zeros((vocab_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def dtanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, inputs, h_prev, c_prev):
        x_s, h_s, c_s, i_s, f_s, o_s, g_s = {}, {}, {}, {}, {}, {}, {}

        h_s[-1] = h_prev
        c_s[-1] = c_prev

        for t, idx in enumerate(inputs):
            x = self.E[idx].reshape(-1, 1)
            x_s[t] = x

            i_s[t] = self.sigmoid(self.W_ix @ x + self.W_ih @ h_s[t-1] + self.b_i)
            f_s[t] = self.sigmoid(self.W_fx @ x + self.W_fh @ h_s[t-1] + self.b_f)
            o_s[t] = self.sigmoid(self.W_ox @ x + self.W_oh @ h_s[t-1] + self.b_o)
            g_s[t] = np.tanh(self.W_cx @ x + self.W_ch @ h_s[t-1] + self.b_c)

            c_s[t] = f_s[t] * c_s[t-1] + i_s[t] * g_s[t]
            h_s[t] = o_s[t] * np.tanh(c_s[t])

        y_hat = self.W_y @ h_s[len(inputs)-1] + self.b_y
        return y_hat, (x_s, h_s, c_s, i_s, f_s, o_s, g_s)