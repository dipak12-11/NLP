from D_main_RNN import WordRNN
from C_train_data import X_embed,y,word_to_idx,idx_to_word,data,embedding_dim,embedding_matrix
from E_train_RNN import train
from F_generate import generate_lyrics

input_size = embedding_dim  # Size of the embedding vector
hidden_size = 64
output_size = len(word_to_idx)

model = WordRNN(input_size, hidden_size, output_size)
train(model, data, epochs=80, learning_rate=0.047)


# Generate text
seed = "love"
output = generate_lyrics(model, seed, word_to_idx, idx_to_word,embedding_matrix, length=100)
print("\n🎵 AI Lyrics:\n", output)
