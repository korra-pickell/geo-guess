import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from progress.bar import Bar

coords_file = r'E:\DATA\GEO-GUESS\coordinates\coordinates.txt'
images_dir = r'E:\DATA\GEO-GUESS\images'

LAT_MAX = 48.99894
LAT_MIN = 24.36335
LON_MAX = -66.84510
LON_MIN = -125.12020

TEST_PER = 0.2



def country_bound_check():
    geolocator = Nominatim(user_agent="geoDataCleaning")
    coord_lines = open(coords_file).readlines()

    

    clean_coord_file = open(r'E:\DATA\GEO-GUESS\coordinates\coordinates_bounded.txt','a')

    for index, line in enumerate(coord_lines):
        lat,long = line.split(',')[1:]
        if (LAT_MIN < float(lat) < LAT_MAX) and (LON_MIN < float(long) < LON_MAX):
            clean_coord_file.write(line)


def dist_cleaning():
    coord_filepath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    scaled_file = r'E:\DATA\GEO-GUESS\coordinates\scaled.txt'
    coord_lines = open(coord_filepath).readlines()
    bar = Bar(' PROCESSING ', max = len(coord_lines), suffix='%(percent)d%%')
    for line in coord_lines[:500]:
        path,lat,long,b = line.replace('\n','').split(',')
        s_lat,s_long = (float(lat) - LAT_MIN)/(LAT_MAX - LAT_MIN),(float(long) - LON_MIN)/(LON_MAX - LON_MIN)



def quality_analysis():
    blur_data_file = r'E:\DATA\GEO-GUESS\coordinates\coords_wblur_data.txt'
    bar = Bar(' PROCESSING ', max = len(os.listdir(images_dir)), suffix='%(percent)d%%')
    '''
    for index,impath in enumerate(os.listdir(images_dir)):
        img = cv2.imread(images_dir + '\\' + impath)
        b = round(cv2.Laplacian(img, cv2.CV_64F).var())
        blur_file = open(blur_data_file,'a')
        blur_file.write(str(b)+'\n')
        blur_file.close()
        bar.next()
    '''
    clean_coord_lines = open(r'E:\DATA\GEO-GUESS\coordinates\coordinates_bounded.txt','r').readlines()

    for index,line in enumerate(clean_coord_lines):
        impath,lat,long = line.split(',')
        img = cv2.imread(impath)
        b = round(cv2.Laplacian(img, cv2.CV_64F).var())
        blur_file = open(blur_data_file,'a')
        blur_file.write(impath+','+lat+','+long.replace('\n','')+','+str(b)+'\n')
        blur_file.close()
        bar.next()

    bar.finish()

def image_aug():
    coord_clean_file = r'E:\DATA\GEO-GUESS\coordinates\coords_wblur_data.txt'
    coord_lines = open(coord_clean_file,'r').readlines()
    aug_img_filepath = r'E:\DATA\GEO-GUESS\aug-256\images'
    aug_coord_filepath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    bounding_points = [0,120,240,360,480]
    bound_hw = 745
    MIN_B = 200
    IMG_OUT_SIZE = 512

    aug_img_index  = 0
    bar = Bar(' PROCESSING ', max = len(os.listdir(images_dir)), suffix='%(percent)d%%')
    for index,line in enumerate(coord_lines):
        bar.next()
        impath,lat,long,b = line.replace('\n','').split(',')
        if int(b) > MIN_B:
            img = cv2.imread(impath)
            for p in bounding_points:
                cropped_img = img[:,p:p+bound_hw]
                resized = cv2.resize(cropped_img, (IMG_OUT_SIZE,IMG_OUT_SIZE))
                cv2.imwrite(aug_img_filepath+'\\'+str(aug_img_index)+'.jpg',resized)
                aug_txt = open(aug_coord_filepath,'a')
                aug_txt.write(aug_img_filepath+'\\'+str(aug_img_index)+'.jpg,'+lat+','+long+','+b+'\n')
                aug_img_index += 1
                aug_txt.close()
                #s = input('...')
    bar.finish()

def to_npz():
    coord_filepath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    coord_lines = open(coord_filepath).readlines()
    img_array,coord_array = [],[]
    bar = Bar(' PROCESSING ', max = len(coord_lines), suffix='%(percent)d%%')
    for line in coord_lines[:5000]:
        path,lat,long,b = line.replace('\n','').split(',')
        s_lat,s_long = (float(lat) - LAT_MIN)/(LAT_MAX - LAT_MIN),(float(long) - LON_MIN)/(LON_MAX - LON_MIN)
        img = cv2.resize(cv2.imread(path),(128,128))
        img_array.append(np.array(img)/255.0)
        coord_array.append([s_lat,s_long])
        bar.next()

    img_array = np.array(img_array)
    coord_array = np.array(coord_array)

    

    np.savez(r'E:\DATA\GEO-GUESS\aug-256\compressed\256-M.npz',
                xdata = img_array, 
                ydata = coord_array)
    bar.finish()
    
    

if __name__ == '__main__':
    to_npz()