import metrics
import os
from tensorflow import keras
from matplotlib import pyplot as plt


os.mkdir('weights/img')
os.mkdir('graphs/img')
os.mkdir('info/img')


def run(source):
    import time

    print('')
    print('')
    print('=================================')
    print('=================================')
    print('Starting training sequence for {}'.format(source))
    print('=================================')
    print('=================================')
    print('')
    print('')

    os.mkdir('graphs/img/{}'.format(source))

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

    opt = keras.optimizers.RMSprop()

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy', metrics.f1, metrics.precision, metrics.recall])

    model.summary()

    batch_size = 32

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'data/train_{}'.format(source),
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    start = time.time()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=25,
        verbose=1
    )
    end = time.time()

    model.save_weights('weights/img/{}.h5'.format(source))
    time = 'Total time: ' + str(end - start) + ' seconds'
    f = open('info/img/{}.txt'.format(source), 'w+')
    f.write(time + '\n')
    f.write('Final training accuracy: ' + str(history.history['acc'][-1]) + '\n')
    f.write('Final training loss: ' + str(history.history['loss'][-1]) + '\n')
    f.write('Final training F1: ' + str(history.history['f1'][-1]) + '\n')
    f.write('Final training precision: ' + str(history.history['precision'][-1]) + '\n')
    f.write('Final training recall: ' + str(history.history['recall'][-1]) + '\n')
    f.close()

    plt.plot(history.history['acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('graphs/img/{}/acc.png'.format(source))
    plt.clf()

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('graphs/img/{}/loss.png'.format(source))
    plt.clf()

    plt.plot(history.history['f1'])
    plt.title('Model F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.savefig('graphs/img/{}/f1.png'.format(source))
    plt.clf()

    plt.plot(history.history['precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.savefig('graphs/img/{}/precision.png'.format(source))
    plt.clf()

    plt.plot(history.history['recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.savefig('graphs/img/{}/recall.png'.format(source))
    plt.clf()
    plt.close()


run('manual')
run('machine')
run('crowd')
