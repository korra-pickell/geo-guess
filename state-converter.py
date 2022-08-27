import time, shutil
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


def build_folders():
    c_path = r'E:\DATA\GEO-GUESS\states\coords_states.txt'
    coord_lines = open(c_path).readlines()
    for line in coord_lines:
        parts = line.split(',')
        im_path,state = parts[0],parts[-1]
        print(im_path,state)
        s = input('...')

build_folders()