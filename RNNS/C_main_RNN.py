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
    def forward(self, x, h_prev):
        h_next=np.tanh(np.dot(self.W_x, x) + np.dot(self.W_h, h_prev) + self.b_h)
        y_hat = numpy.dot(self.W_y, h_next) + self.b_y
        p=np.exp(y_hat) / np.sum(np.exp(y_hat))
        return h_next, p
    