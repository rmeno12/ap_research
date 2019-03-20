from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
import sklearn.datasets as skds
from pathlib import Path
import pandas as pd
import numpy as np

np.random.seed(1237)

path_train = 'data/20news-bydate-train'
path_test = 'data/20news-bydate-test'

files_train = skds.load_files(path_train, load_content=False)

label_index_train = files_train.target
label_names_train = files_train.target_names
labelled_files_train = files_train.filenames

data_tags = ['filename', 'category', 'news']
data_list = []

i = 0
for f in labelled_files_train:
    if i % 50:
        print(i, f)
    data_list.append((f, label_names_train[label_index_train[i]], Path(f).read_text(encoding='windows-1252')))
    i += 1

data = pd.DataFrame.from_records(data_list, columns=data_tags)
print("Loaded Data")

train_size = int(len(data) * .8)

train_posts = data['news'][:train_size]
train_tags = data['category'][:train_size]
train_files_names = data['filename'][:train_size]

test_posts = data['news'][train_size:]
test_tags = data['category'][train_size:]
test_files_names = data['filename'][train_size:]

num_labels = 20
vocab_size = 15000
batch_size = 128

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

X_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
X_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_tags)
Y_train = encoder.transform(train_tags)
Y_test = encoder.transform(test_tags)

encoder = LabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

model = keras.models.Sequential()
model.add(keras.layers.Dense(512, input_shape=(vocab_size,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(512))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(num_labels))
model.add(keras.layers.Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

plt.plot(history.history['acc'])
plt.title('Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['val_acc'])
plt.title('Model Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
