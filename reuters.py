# Use PlaidML backend of keras when PlaidML is installed
try:
    import plaidml.keras
    plaidml.keras.install_backend()
except ImportError:
    print("Cannot use PlaidML backend.")
# Use Keras
from keras import models
from keras import layers
from keras.datasets import reuters
import numpy as np
import matplotlib.pyplot as plt

will_print_data = True

# Vectorize data with One-hot
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    print(results)
    return results

# Get Reuters data from Keras datasets and choose 10000 words with high frequency.
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
if will_print_data:
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Get maximum number of words to create input dimension.
max_words = max([max(sequence) for sequence in train_data]) + 1
# Get maximum number of labels to create output dimension.
max_labels = max(train_labels) + 1

# vectorize data for train
x_train = vectorize_sequences(train_data, max_words)
y_train = vectorize_sequences(train_labels, max_labels)
# vectorize data for test
x_test = vectorize_sequences(test_data, max_words)
y_test = vectorize_sequences(test_labels, max_labels)

# Split the x_train into train data and validation data
len_val = int(len(x_train) * 0.4)
x_val = x_train[:len_val]
partial_x_train = x_train[len_val:]
y_val = y_train[:len_val]
partial_y_train = y_train[len_val:]

# Create model
model = models.Sequential()
model.add(layers.Dense(int(max_labels * 1.5), activation='relu', input_shape=(max_words,)))
model.add(layers.Dense(int(max_labels * 1.5), activation='relu'))
model.add(layers.Dense(max_labels, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit
history = model.fit(partial_x_train, partial_y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val))

# Print results
result = model.evaluate(x_test, y_test)
print(result)

# Print history
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
