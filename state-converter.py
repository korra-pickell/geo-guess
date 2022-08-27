import time, shutil, os
from geopy.geocoders import Nominatim
from progress.bar import Bar


def get_states():
    geolocator = Nominatim(user_agent="geoDataCleaning")

    cpath = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
    save_path = r'E:\DATA\GEO-GUESS\states\coords_states.txt'

    coord_lines = open(cpath).readlines()

    bar = Bar(' PROCESSING ', max = len(coord_lines), suffix='%(percent)d%%')
    current_state = ''
    for index,line in enumerate(coord_lines[59415:]):
        bar.next()
        if index%5 == 0:
            lat,lon = line.split(',')[1],line.split(',')[2]
            #print(lat,lon)
            location = geolocator.reverse(lat+','+lon)
            current_state = location.raw['address']['state']
            time.sleep(1)
        dfile = open(save_path,'a')
        dfile.write(line.replace('\n','')+','+current_state+'\n')
        dfile.close()
    bar.finish()

def check_folder_exists(state):
    parent = r'E:\DATA\GEO-GUESS\states\Image Classes'
    subfolders = os.listdir(parent)
    return state in subfolders

def build_folders():
    c_path = r'E:\DATA\GEO-GUESS\states\coords_states.txt'
    coord_lines = open(c_path).readlines()
    bar = Bar(' PROCESSING ', max = len(coord_lines), suffix='%(percent)d%%')
    for line in coord_lines:
        bar.next()
        parts = line.split(',')
        im_path_src,state = parts[0],parts[-1].strip('\n')
        img_name = parts[0].split('\\')[-1]
        im_path_tar = os.path.join(r'E:\DATA\GEO-GUESS\states\Image Classes',state,img_name)
        
        if check_folder_exists(state):
            shutil.copy(im_path_src,im_path_tar)
        else:
            try: 
                os.mkdir(os.path.join(r'E:\DATA\GEO-GUESS\states\Image Classes',state))
                shutil.copy(im_path_src,im_path_tar)
            except OSError:
                print("couldn't make folder "+state)
    bar.finish()

build_folders()