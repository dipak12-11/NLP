import numpy as np

def generate_lyrics_lstm(model, seed_word, word_to_idx, idx_to_word, length=50):
    hidden_size = model.hidden_dim
    vocab_size = len(word_to_idx)

    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    generated = [seed_word]
    word_idx = word_to_idx.get(seed_word.lower(), 0)
    x = model.E[word_idx].reshape(-1, 1)

    for _ in range(length):
        # LSTM gate calculations
        i = model.sigmoid(model.W_ix @ x + model.W_ih @ h + model.b_i)
        f = model.sigmoid(model.W_fx @ x + model.W_fh @ h + model.b_f)
        o = model.sigmoid(model.W_ox @ x + model.W_oh @ h + model.b_o)
        g = np.tanh(model.W_cx @ x + model.W_ch @ h + model.b_c)

        # Update cell and hidden states
        c = f * c + i * g
        h = o * np.tanh(c)

        # Output layer
        y = model.W_y @ h + model.b_y
        y -= np.max(y)  # softmax stability
        p = np.exp(y) / np.sum(np.exp(y))

        idx = np.random.choice(range(vocab_size), p=p.ravel())
        word = idx_to_word[idx]
        generated.append(word)

        # Update input for next step
        x = model.E[idx].reshape(-1, 1)

    return ' '.join(generated)
