import numpy as np
def train(model, data, epochs=5, learning_rate=0.01):
    for epoch in range(epochs):
        total_loss = 0
        h_prev = np.zeros((model.hidden_dim, 1))
        c_prev = np.zeros((model.hidden_dim, 1))

        for x_seq, y_seq in data:
            cache = model.forward(x_seq, h_prev, c_prev)
            loss, dW_ix, dW_ih, dW_fx, dW_fh, dW_ox, dW_oh, dW_cx, dW_ch,dW_y ,db_i, db_f, db_o, db_c, db_y = model.backward(x_seq, y_seq,cache)

            # Update weights
            model.W_ix -= learning_rate * dW_ix
            model.W_ih -= learning_rate * dW_ih
            model.W_fx -= learning_rate * dW_fx
            model.W_fh -= learning_rate * dW_fh
            model.W_ox -= learning_rate * dW_ox
            model.W_oh -= learning_rate * dW_oh
            model.W_cx -= learning_rate * dW_cx
            model.W_ch -= learning_rate * dW_ch
            model.b_i -= learning_rate * db_i
            model.b_f -= learning_rate * db_f
            model.b_o -= learning_rate * db_o
            model.b_c -= learning_rate * db_c
            model.W_y -= learning_rate * dW_y
            model.b_y -= learning_rate * db_y

            total_loss += loss

        print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
