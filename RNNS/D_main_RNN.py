import numpy 
def one_hot(index,size):
    """
    Create a one-hot encoded vector of a given size with a 1 at the specified index.
    
    Args:
        index (int): The index to set to 1 in the one-hot vector.
        size (int): The size of the one-hot vector.
        
    Returns:
        numpy.ndarray: A one-hot encoded vector of the specified size.
    """
    vector = numpy.zeros(size, dtype=int)
    vector[index] = 1
    return vector

class WordRNN:
   def __init__(self,input_size,hidden_size,output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.W_h=numpy.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x=numpy.random.randn(input_size, hidden_size) * 0.01 #embedding matrix
        self.W_y=numpy.random.randn(hidden_size, output_size) * 0.01
        self.b_h=numpy.zeros((hidden_size, 1))
        self.b_y=numpy.zeros((output_size, 1))
   def forward(self, x_seq, h_prev):
        hs,ps={},{}
        hs[-1]=h_prev
        for t in range(1):
            x_t=one_hot(x_seq[t],self.W_x.shape[0]).reshape(-1, 1)  # shape: (vocab_size, 1)
            # print("x_t shape:", x_t.shape)
            # print("hs[t-1] shape:", hs[t-1].shape, "t:", t)
            h_next=numpy.tanh(numpy.dot(self.W_x.T, x_t) + numpy.dot(self.W_h, hs[t-1]) + self.b_h)
            # print("h_next shape:", h_next.shape)
            y_hat = numpy.dot(self.W_y.T, h_next) + self.b_y
            p=numpy.exp(y_hat) / numpy.sum(numpy.exp(y_hat))
            hs[t] = h_next
            ps[t] = p
            # print("asdfsdf",ps[t].shape)
            # print("hs[t] shape:", hs[t].shape)
            # print("p shape:", p.shape)
        return hs,ps 
    
   def backward(self, x_seq, y_target, hs, ps):
    dWxh = numpy.zeros_like(self.W_x)
    dWhh = numpy.zeros_like(self.W_h)
    dWhy = numpy.zeros_like(self.W_y)
    dbh  = numpy.zeros_like(self.b_h)
    dby  = numpy.zeros_like(self.b_y)
    dh_next = numpy.zeros_like(hs[0])
    loss = 0

    # We compute loss and gradient only from the **last time step** (standard for word-level RNNs)
    t = len(x_seq) - 1  # last time step
    dy = numpy.copy(ps[t])  # shape: (vocab_size, 1)

    if dy.ndim == 1:
        dy = dy.reshape(-1, 1)

    target_idx = y_target if isinstance(y_target, int) else y_target[0]
    dy[target_idx, 0] -= 1  # gradient of softmax + cross-entropy

    loss += -numpy.log(ps[t][target_idx, 0] + 1e-9)  # scalar loss (add epsilon for stability)

    dWhy += numpy.dot(hs[t], dy.T)  # (hidden, 1) x (1, vocab) = (hidden, vocab)
    # print("dby shape:", dby.shape)
    dby  += dy

    # Backprop through time for hidden state
    dh = numpy.dot(self.W_y, dy) + dh_next  # shape: (hidden, 1)
    # print("dh shape:", dh.shape)

    for t in reversed(range(1)):
        dh_raw = (1 - hs[t] ** 2) * dh
        # print("dh_raw shape:", dh_raw.shape)

        x_t = one_hot(x_seq[t], self.W_x.shape[0]).reshape(-1, 1)  # vocab_size
        # print("x_t shape:", x_t.shape)

        dWxh += numpy.dot(x_t, dh_raw.T)  # (vocab, 1) x (1, hidden) = (vocab, hidden)
        dWhh += numpy.dot(dh_raw, hs[t - 1].T) if t != 0 else 0
        dbh  += dh_raw

        dh = numpy.dot(self.W_h.T, dh_raw)

    # Clip to prevent exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        numpy.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby