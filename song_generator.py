import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Modeli ve tokenizer'ı yükle
model = load_model('turkish_song_generator.keras')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
max_sequence_len = np.load('max_sequence_len.npy')

print("Model başarıyla yüklendi.")

def generate_song(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = predicted.flatten()
        
        # Top-k sampling (k=10)
        temperature = 1.5
        predicted = np.log(predicted + 1e-7) / temperature
        predicted = np.exp(predicted)
        predicted = predicted / np.sum(predicted)
        chosen_index = np.random.choice(len(predicted), p=predicted)
        
        next_word = tokenizer.index_word.get(chosen_index, '')
        if next_word == '':
            break
        seed_text += " " + next_word
    return seed_text

# Örnek kullanım
seed_text = "Sevgilim"  # veya "Gece", "Yalnızlık", "Mutluluk"
print(generate_song(seed_text, next_words=100))