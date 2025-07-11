import numpy as np

def generate_lyrics(model, seed_word, word_to_ix, ix_to_word, embedding_matrix, length=50):
    hidden_size = model.hidden_size

    x = word_to_ix.get(seed_word.lower(), 0)
    h = np.zeros((hidden_size, 1))
    generated = [seed_word]

    for _ in range(length):
        x_t = embedding_matrix[x].reshape(-1, 1)  # Use GloVe embedding instead of one-hot
        h = np.tanh(np.dot(model.W_h, h) + np.dot(x_t.T, model.W_in).T+ model.b_h)
        y = np.dot(model.W_y.T, h) + model.b_y
        p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))

        x = np.random.choice(range(len(word_to_ix)), p=p.ravel())
        next_word = ix_to_word[x]
        generated.append(next_word)

    return ' '.join(generated)



# h_next = np.tanh(np.dot(self.W_in, x_t) + np.dot(self.W_h, hs[t - 1]) + self.b_h)