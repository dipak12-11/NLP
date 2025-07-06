import numpy as np
def train(model,data,epochs=5, learning_rate=0.01):
    h_prev=np.zeros((model.hidden_size,1))
    for epoch in range(epochs):
        total_loss=0
        for x_seq, y_seq in data:
            # Forward pass
            hs, ps = model.forward(x_seq, h_prev)
            
            # Backward pass
            loss, dW_x, dW_h, dW_y, db_h, db_y = model.backward(x_seq, y_seq, hs, ps)
            
            # Update weights
            model.W_x -= learning_rate * dW_x
            model.W_h -= learning_rate * dW_h
            model.W_y -= learning_rate * dW_y
            model.b_h -= learning_rate * db_h
            model.b_y -= learning_rate * db_y
            
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")