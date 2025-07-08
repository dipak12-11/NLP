with open('charlie_cleaned_lyrics.txt', "r", encoding='utf-8') as f:
    text = f.read().lower()
text = text.replace('\n', ' ').replace('\r', ' ')
# print(text[:1000])  # Print the first 1000 characters to verify the content
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens[:100])  # Print the first 100 tokens to verify tokenization

seq_len=1
input_sequences = []
target=[]
for i in range(len(tokens)-seq_len):
    input_sequences.append(tokens[i:i+seq_len])
    target.append(tokens[i+seq_len])
    
# print("Input:", input_sequences[0])
# print("Target:", target[0])

vocab = sorted(set(tokens))
# print("Vocabulary size:", len(vocab))
word_to_idx={word:idx for idx,word in enumerate(vocab)}
idx_to_word={idx:word for idx,word in enumerate(vocab)}

x = [[word_to_idx[word] for word in seq] for seq in input_sequences]
y = [word_to_idx[word] for word in target]

# print("Input sequence example:", x[0:3])
# print("Target word example:", y[0:3])

data=list(zip(x, y))
# print(data[0:3])  # Print the first 3 examples to verify the data structure                                                 
# print("Number of training examples:", len(data))
# print("Vocabulary size:", len(word_to_idx))
# print(len(vocab))