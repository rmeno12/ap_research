import os
import pandas as pd
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize


manual_data = pd.read_csv('manual_train.csv')
crowdsource_data = pd.read_csv('crowdsource_train.csv')
machine_data = pd.read_csv('machine_train.csv')


def load_format_image(url):
    image = io.imread(url)

    if image[1].dtype != 'uint8':
        image = image[0]

    if len(image.shape) == 3:
        image = rgb2gray(image)
    else:
        image = image / 255.0

    image = resize(image, (200, 200))

    return image


def save_data(data_inp, dir_head):
    # os.rmdir('{}_dogs'.format(dir_head))
    # os.rmdir('{}_cats'.format(dir_head))
    os.mkdir('{}_dogs'.format(dir_head))
    os.mkdir('{}_cats'.format(dir_head))
    for row in range(len(data_inp.index)):
        path = 'cats' if data_inp.iat[row, 1] == 1 else 'dogs'
        io.imsave("{}_{}/{}.jpeg".format(dir_head, path, str(row)), load_format_image(data_inp.iat[row, 2]))

        if row % 10 == 0:
            print("Done with row", row)


save_data(machine_data, 'machine')
