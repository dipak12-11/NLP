import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np  
with open('charlie_cleaned_lyrics.txt', "r", encoding='utf-8') as f:
    text = f.read().lower()
text = text.replace('\n', ' ').replace('\r', ' ')
tokens = word_tokenize(text)


seq_len=1
input_sequences = []
target=[]
for i in range(len(tokens)-seq_len):
    input_sequences.append(tokens[i:i+seq_len])
    target.append(tokens[i+seq_len])

vocab = sorted(set(tokens))
word_to_idx={word:idx for idx,word in enumerate(vocab)}
idx_to_word={idx:word for idx,word in enumerate(vocab)}
vocab_size = len(vocab)

file_path = r"D:\ML\learn_and_practice\glove.6B.50d.txt"

def load_glove_embeddings(file_path):
    glove={}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = np.array(values[1:], dtype=np.float32)
            glove[word] = vec
    return glove
embedding_dim=50
glove=load_glove_embeddings(file_path)

embedding_matrix = np.zeros((vocab_size, embedding_dim))
oov_count = []
for word in vocab:
    if word in glove:
        embedding_matrix[word_to_idx[word]] = glove[word]
    else:
        embedding_matrix[word_to_idx[word]] = np.random.normal(scale=0.6, size=(embedding_dim,))
        oov_count.append(word)
print(f"Number of OOV words: {(oov_count)}")


X_embed=[]
for seq in input_sequences:
    embed_seq=[embedding_matrix[word_to_idx[word]] for word in seq]
    X_embed.append(np.array(embed_seq))

y = [word_to_idx[word] for word in target]


data=list(zip(X_embed, y))