import numpy as np

def train(model, data, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        h_prev = np.zeros((model.hidden_size, 1))

        for x_seq, y_seq in data:
            hs, ps = model.forward(x_seq, h_prev)
            loss,dw_in, dWhh, dWhy, dbh, dby = model.backward(x_seq, y_seq, hs, ps)

            # Update weights
            # model.W_x -= learning_rate * dWx
            model.w_in -= learning_rate * dw_in
            model.W_h -= learning_rate * dWhh
            model.W_y -= learning_rate * dWhy
            model.b_h -= learning_rate * dbh
            model.b_y -= learning_rate * dby

            total_loss += loss

        # avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
