# Take trained model and display the difference between test points and predictions


import cv2
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


LAT_MAX = 48.99894
LAT_MIN = 24.36335
LON_MAX = -66.84510
LON_MIN = -125.12020


def get_model():
    model = tf.keras.models.load_model(r'E:\models\geo-guess-model.h5')
    return model


def get_demo_images():
    cpath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    coord_lines = open(cpath).readlines()

    demo_lines = coord_lines[-50:][::-5]

    img_arr = []
    coords = []
    for line in demo_lines:
        im_path = line.split(',')[0]
        img = cv2.resize(cv2.imread(im_path),(128,128))
        img_arr.append(np.array(img)/255.0)

        lat,lon = line.split(',')[1], line.split(',')[2]
        coords.append([lat,lon])

    return img_arr, coords
        


def get_map():
    m = Basemap(projection='mill',
                llcrnrlat=25,
                llcrnrlon=-130,
                urcrnrlat=50,
                urcrnrlon=-60,
                resolution='l')

    m.drawcoastlines(linewidth=0.5,color='gray')
    m.drawcountries(linewidth=0.5,color='gray')
    m.drawstates(color='gray')

    return m


def scale_coords(lat,lon):
    s_lat = ((LAT_MAX-LAT_MIN) * lat) + LAT_MIN
    s_lon = ((LON_MAX-LON_MIN) * lon) + LON_MIN

    return s_lat,s_lon


def main():
    model = get_model()
    images,coords = get_demo_images()
    m = get_map()

'''
m = get_map()
nylat,nylon = 40.7,-74

xpt,ypt = m(nylat,nylon)
m.plot(xpt,ypt,'co')
plt.show()
'''

get_demo_images()