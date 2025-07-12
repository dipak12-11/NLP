from D_main_RNN import WordRNN
from C_train_data import data, vocab_size, word_to_idx, idx_to_word, embedding_matrix
from E_train_RNN import train
from F_generate import generate_lyrics

embedding_dim = 50
hidden_size = 64
output_size = len(word_to_idx)

model = WordRNN( embedding_dim, hidden_size, vocab_size)
train(model, data, epochs=50, learning_rate=0.047)

# Generate text
seed = "love"
output = generate_lyrics(model, seed, word_to_idx, idx_to_word,embedding_matrix,length=100)
print("\n🎵 AI Lyrics:\n", output)
