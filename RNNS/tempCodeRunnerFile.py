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
        self.W_h=numpy.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x=numpy.random.randn(input_size, hidden_size) * 0.01 #embedding matrix
        self.W_y=numpy.random.randn(hidden_size, output_size) * 0.01
        self.b_h=numpy.zeros((hidden_size, 1))
        self.b_y=numpy.zeros((output_size, 1))
    def forward(self, x_seq, h_prev):
        hs,ps={},{}
        hs[-1]=h_prev
        for t in range(len(x_seq)):
            x_t=one_hot(x[t],self.W_x.shape[0]).reshape(-1,1)
            h_next=np.tanh(np.dot(self.W_x, x) + np.dot(self.W_h, h_prev) + self.b_h)
            
            y_hat = numpy.dot(self.W_y, h_next) + self.b_y
            p=np.exp(y_hat) / np.sum(np.exp(y_hat))
            hs[t] = h_next
            ps[t] = p
        return hs,ps 
    
    def backward(self, x_seq, y_seq, hs, ps):
        dWxh=np.zeros_like(self.W_x)
        dWhh=np.zeros_like(self.W_h)
        dWhy=np.zeros_like(self.W_y)
        dbh=np.zeros_like(self.b_h)     
        dby=np.zeros_like(self.b_y)
        dh_next = numpy.zeros_like(hs[0])
        loss = 0
        for t in reversed(range(len(x_seq))):
             # Step 1: dL/do = softmax - one-hot
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1    # Shape: (output_size, 1)
        loss += -np.log(ps[t][targets[t]])

        # Step 2: Gradients for output weights
        dW_y += np.dot(dy, hs[t].T)    # (O, 1) · (1, H) → (O, H)
        db_y += dy                     # (O, 1)

        # Step 3: Backprop into h_t
        dh = np.dot(self.W_y.T, dy) + dh_next    # (H, 1)
        dh_raw = (1 - hs[t] ** 2) * dh           # tanh'

        # Step 4: Gradients for hidden weights and input weights
        x_t = one_hot(x_seq[t], self.W_x.shape[1]).reshape(-1, 1)  # (V, 1)

        dW_x += np.dot(dh_raw, x_t.T)            # (H, 1) · (1, V) → (H, V)
        dW_h += np.dot(dh_raw, hs[t-1].T)        # (H, 1) · (1, H) → (H, H)
        db_h += dh_raw                           # (H, 1)

        # Step 5: Propagate to previous time step
        dh_next = np.dot(self.W_h.T, dh_raw)     # (H, 1)

    # Step 6: Gradient Clipping
    for dparam in [dW_x, dW_h, dW_y, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dW_x, dW_h, dW_y, db_h, db_y
            
    