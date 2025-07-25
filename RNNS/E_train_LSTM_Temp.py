import numpy as np
def train(model, data, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        for X_seq,Y_seq in data:
            h_prev=np.zeros((model.hidden_dim, 1))
            c_prev=np.zeros((model.hidden_dim, 1))
            