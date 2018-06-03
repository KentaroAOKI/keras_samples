# Use PlaidML backend of keras when PlaidML is installed
try:
    import plaidml.keras
    plaidml.keras.install_backend()
except ImportError:
    print("Cannot use PlaidML backend.")
# Use Keras
from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt

will_print_data = True

# Vectorize data with One-hot
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Get IMDb data from Keras datasets and choose 10000 words with high frequency.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
if will_print_data:
    print(max([max(sequence) for sequence in train_data]))
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decode_review)

x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')

# Split the x_train into train data and validation data
len_val = int(len(x_train) * 0.4)
x_val = x_train[:len_val]
partial_x_train = x_train[len_val:]
y_val = y_train[:len_val]
partial_y_train = y_train[len_val:]

# Create model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Fit
history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))

# Print results
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
