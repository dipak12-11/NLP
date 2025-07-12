
import numpy
class WordRNN:
   def __init__(self,input_size,hidden_size,output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.W_h=numpy.random.randn(hidden_size, hidden_size) * 0.01
        self.w_in=numpy.random.randn(input_size, hidden_size) * 0.01
        self.W_y=numpy.random.randn(hidden_size, output_size) * 0.01
        self.b_h=numpy.zeros((hidden_size, 1))
        self.b_y=numpy.zeros((output_size, 1))
   def forward(self, x_seq, h_prev):
        hs,ps={},{}
        hs[-1]=h_prev
        for t in range(len(x_seq)):
            x_t=x_seq[t].reshape(-1, 1)
            h_next=numpy.tanh(numpy.dot(self.w_in.T, x_t) + numpy.dot(self.W_h, hs[t-1]) + self.b_h)
            y_hat = numpy.dot(self.W_y.T, h_next) + self.b_y
            p=numpy.exp(y_hat) / numpy.sum(numpy.exp(y_hat))
            hs[t] = h_next
            ps[t] = p
        return hs,ps 
   def backward(self, x_seq, y_target, hs, ps):
    dWin = numpy.zeros_like(self.w_in)
    dWhh = numpy.zeros_like(self.W_h)
    dWhy = numpy.zeros_like(self.W_y)
    dbh  = numpy.zeros_like(self.b_h)
    dby  = numpy.zeros_like(self.b_y)
    dh_next = numpy.zeros_like(hs[0])
    loss = 0
    t = len(x_seq) - 1  
    dy = numpy.copy(ps[t]) 
    if dy.ndim == 1:
        dy = dy.reshape(-1, 1)
    target_idx = y_target if isinstance(y_target, int) else y_target[0]
    dy[target_idx, 0] -= 1  
    loss += -numpy.log(ps[t][target_idx, 0] + 1e-9)  
    dWhy += numpy.dot(hs[t], dy.T)     
    dby  += dy
    dh = numpy.dot(self.W_y, dy) + dh_next
    for t in reversed(range(1)):
        dh_raw = (1 - hs[t] ** 2) * dh
        x_t = x_seq[t].reshape(-1, 1)    
        dWin += numpy.dot(x_t, dh_raw.T)  
        dWhh += numpy.dot(dh_raw, hs[t - 1].T) if t != 0 else 0
        dbh  += dh_raw
        dh = numpy.dot(self.W_h.T, dh_raw)
    for dparam in [dWin, dWhh, dWhy, dbh, dby]:
        numpy.clip(dparam, -5, 5, out=dparam)

    return loss, dWin, dWhh, dWhy, dbh, dby