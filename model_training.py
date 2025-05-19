import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import load_npz
import pickle

# Hazırlanan veriyi yükle
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train_sparse = load_npz('y_train.npz')
y_test_sparse = load_npz('y_test.npz')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
max_sequence_len = np.load('max_sequence_len.npy')

# Sparse matrisleri dense matrislere dönüştürmeden kullanmak için y_train ve y_test'i sıkıştırılmış dizilere dönüştürelim
y_train = y_train_sparse.argmax(axis=1).A1
y_test = y_test_sparse.argmax(axis=1).A1

# Modeli oluşturun
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))  # Daha büyük embedding boyutu
model.add(LSTM(64, return_sequences=True))                                    # Daha fazla LSTM birimi
model.add(Dropout(0.3))
model.add(LSTM(64))                                                           # Daha fazla LSTM birimi
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))

# Modeli derleyin
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitin (daha fazla epoch ve uygun batch size)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stop], batch_size=128)

# Modeli Keras formatında kaydedin
model.save('turkish_song_generator.keras')

print("Model eğitimi tamamlandı ve kaydedildi.")