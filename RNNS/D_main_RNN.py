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
            x_t=one_hot(x_seq[t],self.W_x.shape[0]).reshape(-1,1)
            h_next=numpy.tanh(numpy.dot(self.W_x.T, x_t) + numpy.dot(self.W_h, hs[t-1]) + self.b_h)
            
            y_hat = numpy.dot(self.W_y.T, h_next) + self.b_y
            p=numpy.exp(y_hat) / numpy.sum(numpy.exp(y_hat))
            hs[t] = h_next
            ps[t] = p
        return hs,ps 
    
    def backward(self, x_seq, y_seq, hs, ps):
     dWxh = numpy.zeros_like(self.W_x)
     dWhh = numpy.zeros_like(self.W_h)
     dWhy = numpy.zeros_like(self.W_y)
     dbh  = numpy.zeros_like(self.b_h)
     dby  = numpy.zeros_like(self.b_y)
     dh_next = numpy.zeros_like(hs[0])
     loss = 0

     for t in reversed(range(len(x_seq))):
        dy = numpy.copy(ps[t])
        dy = numpy.copy(ps[t])
        if dy.ndim == 1:
           dy = dy.reshape(-1, 1)
        target_idx = y_seq[t][0] if isinstance(y_seq[t], (list, numpy.ndarray)) else y_seq[t]
        dy[target_idx, 0] -= 1
        
        loss += -numpy.log(ps[t][y_seq[t]])

        dWhy += numpy.dot(dy, hs[t].T)
        dby  += dy

        dh = numpy.dot(self.W_y.T, dy) + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh

        x_t = one_hot(x_seq[t], self.W_x.shape[1]).reshape(-1, 1)

        dWxh += numpy.dot(dh_raw, x_t.T)
        dWhh += numpy.dot(dh_raw, hs[t - 1].T)
        dbh  += dh_raw

        dh_next = numpy.dot(self.W_h.T, dh_raw)

     for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        numpy.clip(dparam, -5, 5, out=dparam)

     return loss, dWxh, dWhh, dWhy, dbh, dby
