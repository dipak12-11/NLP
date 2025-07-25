import numpy as np
class WordLSTM:
   def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.01,embedding_matrix=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Embedding matrix
        self.E = embedding_matrix if embedding_matrix is not None else np.random.randn(vocab_size, embedding_dim) * 0.01
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
        ps={}

        h_s[-1] = h_prev
        c_s[-1] = c_prev

        for t, idx in enumerate(inputs):
            # print(f"Processing timestep {t} with input index {idx}")
            x = inputs.reshape(-1, 1)
            x_s[t] = x
            # LSTM gate calculations
            # print(self.W_ix.shape, x.shape)
            i_s[t] = self.sigmoid(self.W_ix @ x + self.W_ih @ h_s[t-1] + self.b_i)
            f_s[t] = self.sigmoid(self.W_fx @ x + self.W_fh @ h_s[t-1] + self.b_f)
            o_s[t] = self.sigmoid(self.W_ox @ x + self.W_oh @ h_s[t-1] + self.b_o)
            g_s[t] = np.tanh(self.W_cx @ x + self.W_ch @ h_s[t-1] + self.b_c)

            c_s[t] = f_s[t] * c_s[t-1] + i_s[t] * g_s[t]
            h_s[t] = o_s[t] * np.tanh(c_s[t])

            y_hat = self.W_y @ h_s[t] + self.b_y
            p = np.exp(y_hat) / np.sum(np.exp(y_hat))
            ps[t] = p
        return (ps,x_s, h_s, c_s, i_s, f_s, o_s, g_s)
    
   def backward(self, X_seq, Y_seq, cache):
    ps, x_s, h_s, c_s, i_s, f_s, o_s, g_s = cache

    # Gradients
    dW_ix = np.zeros_like(self.W_ix)
    dW_ih = np.zeros_like(self.W_ih)
    dW_fx = np.zeros_like(self.W_fx)
    dW_fh = np.zeros_like(self.W_fh)
    dW_ox = np.zeros_like(self.W_ox)
    dW_oh = np.zeros_like(self.W_oh)
    dW_cx = np.zeros_like(self.W_cx)
    dW_ch = np.zeros_like(self.W_ch)
    dW_y  = np.zeros_like(self.W_y)
    db_i = np.zeros_like(self.b_i)
    db_f = np.zeros_like(self.b_f)
    db_o = np.zeros_like(self.b_o)
    db_c = np.zeros_like(self.b_c)
    db_y = np.zeros_like(self.b_y)

    dh_next = np.zeros_like(h_s[0])
    dc_next = np.zeros_like(c_s[0])

    loss = 0

    for t in reversed(range(len(X_seq))):
        y_target= Y_seq if isinstance(Y_seq, int) else Y_seq[0]
        dy = np.copy(ps[t])
        dy[y_target] -= 1
        loss += -np.log(ps[t][y_target] + 1e-9)

        # Output layer gradients
        dW_y += dy @ h_s[t].T
        db_y += dy

        # Backprop into h and c
        dh = self.W_y.T @ dy + dh_next
        dc = dc_next + dh * o_s[t] * self.dtanh(c_s[t])

        do_raw = self.dsigmoid(o_s[t]) * (dh * np.tanh(c_s[t]))
        di_raw = self.dsigmoid(i_s[t]) * (dc * g_s[t])
        df_raw = self.dsigmoid(f_s[t]) * (dc * c_s[t-1])
        dg_raw = self.dtanh(g_s[t])     * (dc * i_s[t])

        # Param gradients
        dW_ix += di_raw @ x_s[t].T
        dW_ih += di_raw @ h_s[t-1].T
        db_i += di_raw

        dW_fx += df_raw @ x_s[t].T
        dW_fh += df_raw @ h_s[t-1].T
        db_f += df_raw

        dW_ox += do_raw @ x_s[t].T
        dW_oh += do_raw @ h_s[t-1].T
        db_o += do_raw

        dW_cx += dg_raw @ x_s[t].T
        dW_ch += dg_raw @ h_s[t-1].T
        db_c += dg_raw

        # Gradient for next timestep
        dh_next = (
            self.W_ih.T @ di_raw +
            self.W_fh.T @ df_raw +
            self.W_oh.T @ do_raw +
            self.W_ch.T @ dg_raw
        )
        dc_next = f_s[t] * dc

    # Clip all grads
    for dparam in [dW_ix, dW_ih, dW_fx, dW_fh, dW_ox, dW_oh, dW_cx, dW_ch, dW_y, db_i, db_f, db_o, db_c, db_y]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dW_ix, dW_ih, dW_fx, dW_fh, dW_ox, dW_oh, dW_cx, dW_ch, dW_y, db_i, db_f, db_o, db_c, db_y

            
        