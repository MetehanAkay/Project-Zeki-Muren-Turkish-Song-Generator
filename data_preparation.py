import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, save_npz
import pickle
import re

# Veri setini yükle
data = pd.read_csv('zeki.csv')

# Şarkı sözlerini temizleme fonksiyonu
def clean_lyrics(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Yinelenen kelimeleri kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Türkçe karakterleri koruma
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları kaldır
    return text

# Şarkı sözlerini birleştir ve temizle
lyrics = ' '.join(data['lyric'].apply(clean_lyrics).tolist())

# Tokenizer'ı eğit
tokenizer = Tokenizer()
tokenizer.fit_on_texts([lyrics])
total_words = len(tokenizer.word_index) + 1

# Tokenize edilmiş sekanslar oluştur
input_sequences = []
for line in data['lyric'].apply(clean_lyrics):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Sekansları pad et
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Predictors ve labels oluştur
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

# Sparse matris kullanarak labels'ı one-hot encode yap
ys = csr_matrix((len(labels), total_words), dtype=np.float64)
ys[np.arange(len(labels)), labels] = 1

# Veriyi eğitim ve doğrulama setlerine böl
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2)

# Ön işlenmiş veriyi kaydet
np.save('x_train.npy', x_train)
np.save('x_test.npy', x_test)
save_npz('y_train.npz', y_train)  # save_npz fonksiyonunu kullanarak sparse matrisleri kaydet
save_npz('y_test.npz', y_test)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
np.save('max_sequence_len.npy', max_sequence_len)

print("Veri hazırlama başarıyla tamamlandı.")