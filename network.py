import time
from tensorflow import keras
from matplotlib import pyplot as plt

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


batch_size = 32

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'data/train_manual',
    target_size=(200, 200),
    batch_size=batch_size,
    class_mode='binary')


start = time.time()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    verbose=1
)
end = time.time()

model.save_weights('manual_image_weights.h5')
time = 'Total time: ' + str(end-start) + ' seconds'
f = open('manual_info.txt', 'w+')
f.write(time + '\n')
f.write('Final accuracy: ' + str(history.history['acc'][-1]) + '\n')
f.write('Final loss: ' + str(history.history['loss'][-1]) + '\n')
f.close()

plt.plot(history.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
