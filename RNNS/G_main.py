from D_main_RNN import WordRNN
from C_train_data import x,y,word_to_idx,idx_to_word,data
from E_train_RNN import train
from F_generate import generate_lyrics

input_size = len(word_to_idx)
hidden_size = 64
output_size = len(word_to_idx)

model = WordRNN(input_size, hidden_size, output_size)
train(model, data, epochs=80, learning_rate=0.047)


# Generate text
seed = "love"
output = generate_lyrics(model, seed, word_to_idx, idx_to_word, length=100)
print("\n🎵 AI Lyrics:\n", output)
