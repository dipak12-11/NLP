
import numpy as np
from D_main_RNN import one_hot  

def generate_lyrics(model, seed_word, word_to_ix, ix_to_word, length=50):
    vocab_size = len(word_to_ix)
    hidden_size = model.hidden_size

    x = word_to_ix.get(seed_word.lower(), 0)
    h = np.zeros((hidden_size, 1))
    generated = [seed_word]

    for _ in range(length):
        x_t = one_hot(x, vocab_size).reshape(-1, 1)
        h = np.tanh(np.dot(model.W_x.T, x_t) + np.dot(model.W_h, h) + model.b_h)
        y = np.dot(model.W_y.T, h) + model.b_y
        p = np.exp(y) / np.sum(np.exp(y))

        x = np.random.choice(range(vocab_size), p=p.ravel())
        next_word = ix_to_word[x]
        generated.append(next_word)

    return ' '.join(generated)
