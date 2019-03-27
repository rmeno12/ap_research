import numpy as np
import sklearn.datasets as skds
from pathlib import Path
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer

np.random.seed(1237)

path_train = 'data/20news-bydate-train'
path_test = 'data/20news-bydate-test'

files_train = skds.load_files(path_train, load_content=False)
files_test = skds.load_files(path_test, load_content=False)

label_index_train = files_train.target
label_names_train = files_train.target_names
labelled_files_train = files_train.filenames

label_index_test = files_train.target
label_names_test = files_test.target_names
labelled_files_test = files_test.filenames

data_tags = ['filename', 'category', 'news']
data_list_train = []
data_list_test = []

i = 0
for f in labelled_files_train:
    if i % 50 == 0:
        print('Train datapoint', i)
    data_list_train.append((f, label_names_train[label_index_train[i]], Path(f).read_text(encoding='windows-1252')))
    i += 1

i = 0
for f in labelled_files_test:
    if i % 50 == 0:
        print('Test datapoint', i)
    data_list_test.append((f, label_names_test[label_index_test[i]], Path(f).read_text(encoding='windows-1252')))
    i += 1

data_train = pd.DataFrame.from_records(data_list_train, columns=data_tags)
data_test = pd.DataFrame.from_records(data_list_test, columns=data_tags)
print("Loaded Data")

train_size = int(len(data_train) * .8)

train_posts = data_train['news'][:train_size]
train_tags = data_train['category'][:train_size]
train_files_names = data_train['filename'][:train_size]

val_posts = data_train['news'][train_size:]
val_tags = data_train['category'][train_size:]
val_files_names = data_train['filename'][train_size:]

test_posts = data_test['news']
test_tags = data_test['category']
test_files_names = data_test['filename']

num_labels = 20
vocab_size = 15000
batch_size = 128

tokenizer_train = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer_train.fit_on_texts(train_posts)

X_train = tokenizer_train.texts_to_matrix(train_posts, mode='tfidf')
X_val = tokenizer_train.texts_to_matrix(val_posts, mode='tfidf')
X_test = tokenizer_train.texts_to_matrix(test_posts, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_tags)
Y_train = encoder.transform(train_tags)
Y_val = encoder.transform(val_tags)
Y_test = encoder.transform(test_tags)


np.save('data/news-data/X_train', X_train)
np.save('data/news-data/Y_train', Y_train)
np.save('data/news-data/X_val', X_val)
np.save('data/news-data/Y_val', Y_val)
np.save('data/news-data/X_test', X_test)
np.save('data/news-data/Y_test', Y_test)
