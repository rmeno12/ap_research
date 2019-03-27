from tensorflow import keras
import numpy as np
import metrics
import time

X_train = np.load('data/news-data/X_train.npy')
Y_train = np.load('data/news-data/Y_train.npy')
X_val = np.load('data/news-data/X_val.npy')
Y_val = np.load('data/news-data/Y_val.npy')
X_test = np.load('data/news-data/X_test.npy')
Y_test = np.load('data/news-data/Y_test.npy')
print('Data loaded')

vocab_size = 15000
num_labels = 20
batch_size = 100

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
              metrics=['accuracy', metrics.f1, metrics.precision, metrics.recall])

start = time.time()
history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=25,
                    verbose=1)
end = time.time()

model.save_weights('weights/txt/bs.h5')
time = 'Total time: ' + str(end - start) + ' seconds'
f = open('info/txt/bs.txt', 'w+')
f.write(time + '\n')
f.write('Final training accuracy: ' + str(history.history['acc'][-1]) + '\n')
f.write('Final training loss: ' + str(history.history['loss'][-1]) + '\n')
f.write('Final training F1: ' + str(history.history['f1'][-1]) + '\n')
f.write('Final training precision: ' + str(history.history['precision'][-1]) + '\n')
f.write('Final training recall: ' + str(history.history['recall'][-1]) + '\n')
f.close()
