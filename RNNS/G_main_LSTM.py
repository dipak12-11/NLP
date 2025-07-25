from D_main_LSTM import WordLSTM
from C_train_data import data, vocab_size, word_to_idx, idx_to_word, embedding_matrix
from E_train_LSTM import train
from F_generate_LSTM import generate_lyrics_lstm

embedding_size = 50
hidden_size = 64
output_size = len(word_to_idx)
model = WordLSTM(vocab_size,embedding_size, hidden_size)
train(model, data, epochs=10, learning_rate=0.055)
# Generate text
seed = "love"
output = generate_lyrics_lstm(model, seed, word_to_idx, idx_to_word, length=10)
print("\n🎵 AI Lyrics:\n", output)

