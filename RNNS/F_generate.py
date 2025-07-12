import numpy as np
def generate_lyrics(model, seed_word, word_to_idx, idx_to_word, embedding_matrix, length=50):
    hidden_size = model.hidden_size
    vocab_size = len(word_to_idx)

    h = np.zeros((hidden_size, 1))
    generated = [seed_word]

    word_idx = word_to_idx.get(seed_word.lower(), 0)
    x = embedding_matrix[word_idx].reshape(-1, 1)

    for _ in range(length):
        h = np.tanh(np.dot(model.W_h, h) + np.dot(model.w_in.T, x) + model.b_h)
        y = np.dot(model.W_y.T, h) + model.b_y
        p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))

        idx = np.random.choice(range(vocab_size), p=p.ravel())
        word = idx_to_word[idx]
        generated.append(word)

        x = embedding_matrix[idx].reshape(-1, 1)  # next word input
    return ' '.join(generated)
